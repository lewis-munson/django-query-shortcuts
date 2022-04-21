import copy
import inspect
from functools import update_wrapper

from django.db import models
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP


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
