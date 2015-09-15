import collections
import functools
from threading import Lock
from Queue import Queue
from threading import Thread
import time

# //////////////////////////////////////
# Make decorators better
# //////////////////////////////////////

def simple_decorator(decorator):
    '''This decorator can be used to turn simple functions
    into well-behaved decorators, so long as the decorators
    are fairly simple. If a decorator expects a function and
    returns a function (no descriptors), and if it doesn't
    modify function attributes or docstring, then it is
    eligible to use this. Simply apply @simple_decorator to
    your decorator and it will automatically preserve the
    docstring and function attributes of functions to which
    it is applied.'''
    def new_decorator(f):
        g = decorator(f)
        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g.__dict__.update(f.__dict__)
        return g
    # Now a few lines needed to make simple_decorator itself
    # be a well-behaved decorator.
    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    new_decorator.__dict__.update(decorator.__dict__)
    return new_decorator


def test_simple_decorator():
    @simple_decorator
    def my_simple_logging_decorator(func):
        def you_will_never_see_this_name(*args, **kwargs):
            print 'calling {}'.format(func.__name__)
            return func(*args, **kwargs)
        return you_will_never_see_this_name

    @my_simple_logging_decorator
    def double(x):
        'Doubles a number.'
        return 2 * x

    assert double.__name__ == 'double'
    assert double.__doc__ == 'Doubles a number.'
    print double(155)

# ///////////////////////////////////////
# Memorizers
#   Note: I don't have a preference yet.
# ///////////////////////////////////////

class memoized_v1(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

def test_memoized_v1():
    @memoized_v1
    def fibonacci(n):
       "Return the nth fibonacci number."
       if n in (0, 1):
          return n
       return fibonacci(n-1) + fibonacci(n-2)

    print fibonacci(12)


def memoized_v2(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer

def test_memoized_v2():
    @memoized_v2
    def foo(a, b, keyword=None):
        if keyword is not None:
            print keyword
        return a * b

    print foo(2, 4, keyword="Blue")
    print foo('hi', 3, keyword="Red")
    print foo.cache

# ////////////////////////////////////
# Random ones
# ////////////////////////////////////

def addto(instance):
    def decorator(f):
        import types
        f = types.MethodType(f, instance, instance.__class__)
        setattr(instance, f.func_name, f)
        return f
    return decorator

def test_addto():
    class Foo:
        def __init__(self):
            self.x = 42
    foo = Foo()

    @addto(foo)
    def print_x(self):
        print self.x

    foo.print_x()

# /////////////////////////////////////
# Parallel Processing and Threading
# /////////////////////////////////////

def synchronized(lock):
    '''Synchronization decorator.'''

    def wrap(f):
        def new_function(*args, **kw):
            lock.acquire()
            try:
                return f(*args, **kw)
            finally:
                lock.release()
        return new_function
    return wrap

def test_synchronized():

    my_lock = Lock()

    @synchronized(my_lock)
    def critical1(*args):
        # Interesting stuff goes here.
        pass

    @synchronized(my_lock)
    def critical2(*args):
        # Other interesting stuff goes here.
        pass


class asynchronous(object):
    def __init__(self, func):
        self.func = func

        def threaded(*args, **kwargs):
            self.queue.put(self.func(*args, **kwargs))

        self.threaded = threaded

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def start(self, *args, **kwargs):
        self.queue = Queue()
        thread = Thread(target=self.threaded, args=args, kwargs=kwargs);
        thread.start();
        return asynchronous.Result(self.queue, thread)

    class NotYetDoneException(Exception):
        def __init__(self, message):
            self.message = message

    class Result(object):
        def __init__(self, queue, thread):
            self.queue = queue
            self.thread = thread

        def is_done(self):
            return not self.thread.is_alive()

        def get_result(self):
            if not self.is_done():
                raise asynchronous.NotYetDoneException('the call has not yet completed its task')

            if not hasattr(self, 'result'):
                self.result = self.queue.get()

            return self.result

def test_asynchronous():
    # sample usage
    @asynchronous
    def long_process(num):
        time.sleep(10)
        return num * num

    result = long_process.start(12)

    for i in range(20):
        print i
        time.sleep(1)
        if result.is_done():
            print "result {0}".format(result.get_result())

    result2 = long_process.start(13)

    try:
        print "result2 {0}".format(result2.get_result())

    except asynchronous.NotYetDoneException as ex:
        print ex.message

    return

# /////////////////////////////////////
# Singletons
# /////////////////////////////////////

def singleton_v1(cls):
    ''' Use class as singleton. '''

    cls.__new_original__ = cls.__new__

    @functools.wraps(cls.__new__)
    def singleton_new(cls, *args, **kw):
        it =  cls.__dict__.get('__it__')
        if it is not None:
            return it

        cls.__it__ = it = cls.__new_original__(cls, *args, **kw)
        it.__init_original__(*args, **kw)
        return it

    cls.__new__ = singleton_new
    cls.__init_original__ = cls.__init__
    cls.__init__ = object.__init__
    print "wrapping now"
    return cls

def test_singleton_v1():

    @singleton_v1
    class Foo:
        def __new__(cls):
            print "calling new"
            cls.x = 10
            return object.__new__(cls)

        def __init__(self):
            print "calling init"
            assert self.x == 10
            self.x = 15

    print dir(Foo())
    assert Foo().x == 15

    Foo().x = 20
    print Foo().x
    assert Foo().x == 20

def singleton_v2(cls):
    instance = cls()
    instance.__call__ = lambda: instance
    return instance

def test_singleton_v2():
    @singleton_v2
    class Highlander:
        x = 100
        # Of course you can have any attributes or methods you like.

    Highlander() is Highlander() is Highlander #=> True
    id(Highlander()) == id(Highlander) #=> True
    Highlander().x == Highlander.x == 100 #=> True
    Highlander.x = 50
    Highlander().x == Highlander.x == 50 #=> True

if __name__ == "__main__":
    print "Simple Decorator Test"
    test_simple_decorator()

    print "Memoized V1 test (memoized, class version)"
    test_memoized_v1()

    print "Memoized V2 test (memoized, function version)"
    test_memoized_v2()

    print "Testing addto"
    test_addto()

    print "Testing synchronized"
    test_synchronized()

    #print "Testing asynchronous"
    #test_asynchronous()

    # print "Testing Singleton V1"
    # test_singleton_v1()

    print "Testing Singleton V2"
    test_singleton_v2()
