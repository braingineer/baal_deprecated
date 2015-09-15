"""
Storage system for models and work instances in BAAl

@author bcmcmahan

The vaulting mechanism here will be treated as a reverse visitor pattern
In other words, every thing that wants to be vaulted needs to implement a hook
so that vault can get the right things.

top level functions in the vaultee:
vault_items()
    Return the set of things to be vaulted
from_vault()
    A @classmethod that can take those things stored and make an instance

We will call vault.store(obj)
Vault will then call:
storage_items = obj.vault_store()
and shelve them

This will assume read only items, unless specified. If specified, there will be
a sync operation needed to make things felicitous
"""

import shelve
import os
from functools import wraps

curdir = os.path.dirname(os.path.abspath(__file__))
vault_dir = curdir+"/vault"
vault_get = lambda name, **kwargs: shelve.open("%s/%s.baal" % (vault_dir, name),
                                               **kwargs)
vault_index = vault_get("vault_index", writeback=True)

#vault_index = shelve.open("%s/vault_index.baal" % vault_dir, writeback=True)


def vault_check(strname, spacename, extcheckpos, extcheckneg):
    def f_strname(name):
        if not isinstance(name, type("")):
            print("Name must be a string")
            return False

    def f_spacename(name):
        if " " in name:
            print("Use a name without spaces")
            return False

    def f_extcheckpos(name):
        if name in vault_index.keys():
            print("%s exists in the vault." % name)
            print("Name it something else or remove the %s instance" % name)
            return False

    def f_extcheckneg(name):
        if name not in vault_index.keys():
            print("%s does not exist in the vault" % name)
            print("Please consult vault.manifest")

    def inner_decorator(func):
        @wraps(func)
        def func_wrapper(*args,**kwargs):
            name = args[0]  # assumption. maintain in code.
            if strname:
                f_strname(name)
            if spacename:
                f_spacename(name)
            if extcheckpos:
                f_extcheckpos(name)
            if extcheckneg:
                f_extcheckneg(name)
            return func(*args, **kwargs)
        return func_wrapper
    return inner_decorator


@vault_check(1, 1, 1, 0)
def store(name, callback, description=""):
    try:
        storage_items = callback.vault_items()
        box = vault_get(name)
        # box = shelve.open("%s/%s.baal" % name)
        for item_name, obj in storage_items.items():
            box[item_name] = obj
        box.close()
    except KeyError as e:
        raise KeyError("Storage items have badly formed keys. Use strings")
    vault_index[name] = description
    vault_index.sync()

    return True

@vault_check(1, 1, 0, 1)
def retrieve(name, callback):
    box = vault_get(name)
    # box = shelve.open("%s/%s.baal" % name)
    storage_items = box.items()
    box.close()
    return callback.from_vault(storage_items)

@vault_check(1, 1, 0, 1)
def delete(name):
    if not exists(name):
        print "Not in manifest"
        return
    ans = raw_input("Are you sure you want to delete %s? (y/n): " % name)
    if ans.lower() == "y":
        del vault_index[name]
        vault_index.sync()
        print "It is done!"
    else:
        print "Okey dokes"


def manifest(printit=False):
    """
    Get the name and descriptions of everything in the value
    If passed True, will print in format: "Name: %s, Description: %s \n"
    if passed False, will return list in format: [(name, descriptions), ...]

    Future expansions will allow me to limit this to collections or workbenches.
    """
    if printit:
        print("%s" % ("--\n-\n--".join("Name: %s  \t|| Description: %s" %
                                        (name,desc)
                                        for name, desc in vault_index.items())))
    else:
        return vault_index.items()

def exists(name):
    if name in vault_index.keys():
        return True
    return False

@vault_check(1, 1, 0, 1)
def update(name, callback, description=None):
    if description is None:
        description = vault_index[name]
    return store(name, callback, description)
