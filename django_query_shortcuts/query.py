import copy
import inspect
import re
from functools import reduce
from functools import update_wrapper
from operator import add
from operator import and_
from operator import ior

from django.contrib.postgres.search import SearchHeadline
from django.contrib.postgres.search import SearchQuery
from django.contrib.postgres.search import SearchRank
from django.contrib.postgres.search import SearchVector
from django.db import models
from django.db.models import F
from django.db.models import Prefetch
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.utils.html import escape
from django.utils.safestring import SafeData
from django.utils.safestring import mark_safe


class q_shortcut:
    func = None
    obj = None
    type = None

    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func

    def __get__(self, instance, owner=None):
        """Support instance methods."""
        self.obj = instance
        self.type = owner

        return self

    def Q(self, *args, **kwargs):
        return self.func(self.type, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.obj.filter(self.func(self.obj, *args, **kwargs))


def add_prefix(q_obj, prefix):
    """
    Recursively copies the Q object, prefixing all lookup keys.
    The prefix and the existing filter key are delimited by the
    lookup separator __. Use this feature to delegate existing
    query constraints to a related field.
    """
    return Q(
        *(
            add_prefix(child, prefix)
            if isinstance(child, Q)
            else (prefix + LOOKUP_SEP + child[0], child[1])
            for child in q_obj.children
        ),
        _connector=q_obj.connector,
        _negated=q_obj.negated,
    )


def isdecorator(obj):
    return isinstance(obj, q_shortcut)


class QObjManager(models.Manager):
    @classmethod
    def from_queryset(cls, queryset_class, class_name=None):
        manager = super().from_queryset(queryset_class, class_name=class_name)

        methods = inspect.getmembers(queryset_class, predicate=isdecorator)

        for name, method in methods:
            method_copy = copy.deepcopy(method)

            setattr(manager, name, method_copy)

        return manager


class QObjQuerySet(models.QuerySet):
    @classmethod
    def as_manager(cls):
        manager = QObjManager.from_queryset(cls)

        manager._built_with_as_manager = True

        return manager()


class Tokenizer:
    TOKEN_REGEX = re.compile(r"[^\s\"\']+|\"([^\"]*)\"|\'([^\']*)")
    DEFAULT_OPERATOR = '|'
    OPERATORS = (
        '|',
        '+',
        '-',
    )

    def __init__(self, query):
        self.sorted_queries = {}

        hanging_operator = None

        for token in (_.group() for _ in self.TOKEN_REGEX.finditer(query)):
            search_type = 'raw'

            operator = self.DEFAULT_OPERATOR

            if token[0] in self.OPERATORS:
                operator = token[0]

                token = token[1:]

                if len(token.strip()) == 0:
                    hanging_operator = operator

                    continue

            if token.startswith('"') and token.endswith('"'):
                token = token.strip('"')

                search_type = 'phrase'

                if hanging_operator is not None:
                    operator = hanging_operator

            hanging_operator = None

            if search_type == 'raw' and token[-1] == '*':
                token = '{}:*'.format(token[:-1])

            negated = token.startswith('!')

            # remove bad characters
            translation_table = str.maketrans({
                '\\': None,
                '\'': None,
                ':': None,
                '!': None,
                '&': None,
                '|': None,
                '*': None,
                '(': None,
                ')': None,
                '"': None,
            })

            token = token.translate(translation_table)

            # skip if token is now empty
            if len(token) == 0:
                continue

            if negated:
                token = '!{}'.format(token)

            try:
                self.sorted_queries[operator].append((token, search_type))
            except KeyError:
                self.sorted_queries[operator] = [(token, search_type)]

    def iter_tokens(self):
        for operator, tokens in self.sorted_queries.items():
            for token, search_type in tokens:
                yield operator, token, search_type


def build_postgres_search_query(query):
    search_query = None
    negations = []

    for operator, token, search_type in Tokenizer(query).iter_tokens():
        sq = SearchQuery(token, search_type=search_type)

        if operator == "-":
            negations.append(~sq)

            continue

        if search_query is None:
            search_query = sq
        elif operator == "|":
            search_query = search_query | sq
        else:
            search_query = search_query & sq

    if len(negations) > 0:
        negation = reduce(
            and_,
            negations,
        )

        search_query = negation if search_query is None else search_query & negation

    return search_query


class SearchableQuerySet(models.QuerySet):
    SEARCH_FIELDS = ()
    _searched = False
    _related_field_cache = None

    def search(self, query, search_fields=None, annotate_rank=True, annotate_headlines=True):
        if query is None or query.strip() == '':
            return self.all()

        assert not self._searched, 'This QuerySet has already been searched.'

        self._searched = True

        if search_fields is None:
            search_fields = self.SEARCH_FIELDS

        search_query = build_postgres_search_query(query)

        raw_search_fields = []
        related_model_lookups = {}

        for search_field in search_fields:
            if '__' in search_field:
                lookup_parts = search_field.split('__')
                first_lookup = lookup_parts[0]

                related_model_lookups.setdefault(first_lookup, ())
                related_model_lookups[first_lookup] += (
                    '__'.join(lookup_parts[1:]),
                )
            else:
                raw_search_fields.append(search_field)

        self._related_field_cache = {
            related_field: self.get_related_field_queryset(related_field).search(
                query,
                search_fields=related_search_fields,
                annotate_rank=annotate_rank,
                annotate_headlines=annotate_headlines,
            ) for related_field, related_search_fields in related_model_lookups.items()
        }

        filter_conds = [
            *(
                self.write_lookup(related_field)
                for related_field in self._related_field_cache
            )
        ]

        queryset = self

        if len(raw_search_fields) > 0:
            filter_conds.append(Q(search_vector=search_query))

            queryset = queryset.annotate(
                search_vector=reduce(
                    add,
                    [SearchVector(field) for field in raw_search_fields],
                ),
            )

        queryset = queryset.filter(
            reduce(
                ior,
                filter_conds,
            ),
        )

        if len(raw_search_fields) > 0:
            if annotate_rank:
                queryset = queryset.annotate(
                    search_rank=SearchRank(
                        F('search_vector'),
                        search_query,
                        normalization=2,
                    ),
                )

            if annotate_headlines:
                queryset = queryset.annotate(
                    **{
                        '{}_headline'.format(field): SearchHeadline(
                            field,
                            search_query,
                            start_sel='--SEARCH_HIGHLIGHT--',
                            stop_sel='--END_SEARCH_HIGHLIGHT--',
                            short_word=0,
                            highlight_all=True,
                        ) for field in raw_search_fields
                    },
                )

        if len(related_model_lookups) > 0 and annotate_headlines:
            if isinstance(queryset.query.select_related, dict):
                for related_field in related_model_lookups:
                    # pop conflicting select_related since Django will use this over the prefetch
                    queryset.query.select_related.pop(related_field, None)

            queryset = queryset.prefetch_related(
                *[
                    Prefetch(
                        related_field,
                        queryset=self._related_field_cache[related_field],
                        to_attr='{}_search_results'.format(related_field),
                    ) for related_field in self._related_field_cache if self.supports_prefetch(related_field)
                ],
            )

        return queryset

    def get_related_field_queryset(self, related_field):
        return self.model._meta.get_field(related_field).related_model.objects.all()

    def write_lookup(self, related_field):
        return Q(**{
            '{}__in'.format(related_field): [_.id for _ in self._related_field_cache[related_field]],
        })

    def supports_prefetch(self, related_field):
        return True

    def manual_prefetch(self, obj, related_field):
        raise NotImplementedError()

    def _clone(self):
        clone = super()._clone()

        clone._searched = self._searched
        if self._related_field_cache:
            clone._related_field_cache = self._related_field_cache.copy()

        return clone

    def _fetch_all(self):
        super()._fetch_all()

        if not self._searched:
            return

        for res in self._result_cache:
            if hasattr(res, 'search_details'):
                continue

            if not isinstance(res, self.model):
                continue

            res.search_details = {}

            for field in self.SEARCH_FIELDS:
                if '__' in field:
                    related_field = field.split('__')[0]

                    prefetched_related = getattr(res, '{}_search_results'.format(related_field), None)

                    if prefetched_related is None:
                        try:
                            prefetched_related = self.manual_prefetch(res, related_field)
                        except NotImplementedError:
                            pass

                    if prefetched_related:
                        if isinstance(prefetched_related, list):
                            res.search_details[related_field] = [_.search_details for _ in prefetched_related]
                        else:
                            res.search_details[related_field] = prefetched_related.search_details
                            setattr(res, related_field, prefetched_related)
                else:
                    field_headline = '{}_headline'.format(field)

                    if hasattr(res, field_headline):
                        value = getattr(res, field_headline)

                        if value is None:
                            continue

                        if isinstance(value, SafeData):
                            # already cleaned
                            continue

                        if '--SEARCH_HIGHLIGHT--' not in value:
                            # why does it have a value with no matches?
                            delattr(res, field_headline)
                            continue

                        value = escape(value)
                        value = value.replace('--SEARCH_HIGHLIGHT--', '<b class="highlight">')
                        value = value.replace('--END_SEARCH_HIGHLIGHT--', '</b>')
                        value = mark_safe(value)

                        res.search_details[field] = value

                        setattr(res, field_headline, value)

        self._add_to_search_details()

    def _add_to_search_details(self):
        pass
