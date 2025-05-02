---
title: Python OOP Features
date: 2025-04-12T13:47:20Z
lastmod: 2025-04-12T13:47:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: pythonoop.png
categories:
  - python
tags:
  - oop
  - object oriented programming
  - python
  - decorator

# nolastmod: true
draft: false
---

Studying on OOP Features of Python Programming Language

# 1. Class Method for Polymorphism

```python
class Animal:
	def __init__(self, name):
		self.name = name

	def say(self):
		pass

class Cat:
	def __init__(self, name):
		super().__init__(name)

	def say(self):
		print(f"{self.name}, a Cat")

class Dog:
	def __init__(self, name):
		super().__init__(name)

	def say(self):
		print(f"{self.name}, a Dog")

class AnimalContainer:
	def __init__(self):
		self.animals = []

	def create_animals(self, names):
		for name in names:
			animal = Cat(name)
			animal.say()
			self.animals.append(animal)
```

- With the current implementation, if you want to implement `create_animals` function for Dog, you have to create a new class or a new function for Dogs.
- We can solve this using **classmethods** - we can use classmethod to call the class initializer, making it work like a factory method.

```python
class Animal:
	def __init__(self, name):
		self.name = name

	def say(cls):
		pass

	@classmethod
	def create(cls, name):
		pass

class Cat:
	def __init__(self, name):
		super().__init__(name)

	def say(self):
		print(f"{self.name}, a Cat")

	@classmethod
	def create(cls, name):
		return cls(name)

class Dog:
	def __init__(self, name):
		super().__init__(name)

	def say(self):
		print(f"{self.name}, a Dog")

	@classmethod
	def create(cls, name):
		return cls(name)

class AnimalContainer:
	def __init__(self):
		self.animals = []

	def create_animals(self, animal_cls, names):
		for name in names:
			animal = animal_cls(name)
			animal.say()
			self.animals.append(animal)
```

# 2. super()'s C3 Linearization

super() uses C3 Linearization algorithm to ensure parent classes get called only once even in Diamond inheritance situations happen because of multi-inheritance.

```
1) Child Class always before Parent
2) If same parent comes up multiple times, the first appearance gets prioritized
3) All parent classes appear only once in the MRO order.
```

ex) If B, C Inherit from A and D inherits from B, C:

- `super().__init__()` from D calls in `D, B, C, A` order
- **order of class inheritance matters!** -> if `class D(C, B)`, then it's `D, C, B, A` order.

ex2)

```
         A     B
         \   /
          C D
         / \  \
        E   F  G
          \ | /
            H
```

- `super().__init__()` from H calls in `H, E, F, G, C, D, A, B` order.

## super's Implicit Parameters

super() receives two parameters: the class that provides the MRO view, and the instance that will be used to get the MRO view of that class.

- If not given, it is implicitly assigned as `super(__class__, self)`, so almost always we don't have to declare them.

# 3. Dunder methods

## 3-1. call: Giving function-like behavior

- By implementing `__call__(self, ...)`, we can give our class instance a function-like interface

```python
class Animal:
	def __init__(self, name):
		self.name = name

	def __call__(self):
		println(f'animal {self.name}')

cat = Animal('Garfield')

# prints 'animal Garfield'
cat()
```

## 3-2. add, sub: Operator Overloading

- We can make classes implement operator dunder functions to implement operator interfaces.
- they must **return a new object instead of modifying existing object.**
- **Type checking must be done inside class**

```python
class Point:
    def __init__(self, x, y):
		self.x = x
		self.y = y

	def __add__(self, other):
		if hasattr(other, 'x') and hasattr(other, 'y'):
			return Point(self.x + other.x, self.y + other.y)

		return NotImplemented
```

## 3-3. len, iter, next, getitem, setitem, delitem: Container Interface

- There's a lot of dunder methods to define when we create a Container interface:

```
__len__, __iter__, __next__, __getitem__, __setitem__, index, count, ...
```

- To make things easier, we can **make container classes implement collections.abc types to auto generate most of these methods with few required methods.**

```python
from collections.abc import Sequence

# Needs only __getitem__ and __len__
class Friends:
	def __init__(self):
		self.friend_list = []

	def __getitem__(self, idx):
		return self.friend_list[idx]

	def __len__(self):
		return len(self.friend_list)

```

## 3-4. get, set, set_name: Descriptor Methods

These are called **when the class is used as a member of another class.**

```python
from weakref import WeakKeyDictionary

class Grade:
	def __init__(self):
		# erases entry when reference count is 0
		self._values = WeakKeyDictionary()

	def __get__(self, instance, instance_type):
		if instance not in self._values:
			return None

		return self._values[instance]

	def __set__(self, instance, value):
		self._values[instance] = value

class Exam:
	# descriptors must be defined as class attributes
	math_grade = Grade()
	writing_grade = Grade()

	# if instance attribute, exam.__dict__['math_grade'] exists so it doesn't look for Exam.__dict__, so __get__ isn't called

exam = Exam()

# equal to Exam.__dict__['writing_grade'].__set__(4)
exam.writing_grade = 4

# equal to Exam.__dict__['math_grade'].__get__(exam, Exam)
exam.math_grade
```

We can also use `__set_name__` to automatically assign value to the right member name of the instance.

```python
class Grade:
	def __init__(self):
		self.name = None
		self.internal_name = None

	def __set_name__(self, owner, name):
		self.name = name
		self.internal_name = '_' + name

	def __get__(self, instance, instance_type):
		if instance is None:
			return self

		return getattr(instance, self.internal_name, '')

	def __set__(self, instance, value):
		setattr(instance, self.internal_name, value)

class Exam:
	# descriptors must be defined as class attributes
	math_grade = Grade()
	writing_grade = Grade()

exam.writing_grade = 4
exam.math_grade

```

## 3-5. getattribute, getattr, setattr

lazy attribute computation dunder methods.

- getattr: **called only when member doesn't exist**
- getattribute/setattr: **always called for every member reference/assignment**
  - `__getattribute__` **must return AttributeError** for missing member so that `__getattr__` can be called next

make sure `__getattr__` and `__getattribute__` get called on `super` for times it can get caught in infinite loop

```python

class Dog:
	ACTIVITIES = ['run', 'walk', 'ball']

	def __init__(self):
		self.kind = 'dog'
		self.ready_for_activity = False

	@ready_for_activity.setter
	def ready_for_activity(self, value):
		if isinstance(value, bool):
			self.ready_for_activity = meal
		else:
			raise ValueError

	@property
	def ready_for_activity(self):
		return self.ready_for_activity

	def __getattribute__(self, name):
		if self.ready_for_activity:
			return super().__getattribute__(self, name)

		raise AttributeError

	def __getattr__(self, name):
		# called when __getattribute__ raises AttributeError
		if name in ACTIVITIES:
			setattr(self, name, True)
			return True

```

## 3-6. init_subclass

Meta programming can be done with Meta types that inherit `type`, but multiple inheritance of Meta types isn't supported.

Using `__init_subclass()__` solves diamond inheritance problem, and thus supports multiple inheritance

```python
class Filled:
	VALID_COLORS = ['red', 'green', 'blue']
	color = None

	def __init_subclass__(cls):
		super().__init_subclass__()

		if cls.color not in VALID_COLORS:
			raise ValueError(f"Not supported color value: {cls.color}")

class Polygon:
	sides = None

	def __init_subclass__(cls):
		super().__init_subclass__()

		if not isinstance(cls.sides, int) or cls.sides < 3:
			raise ValueError(f"sides {cls.sides} must be an integer greater than 2.")

# valid, so no error
class GreenSquare(Filled, Polygon):
	color = 'green'
	sides = 4

# invalid, throws ValueError("Not supported color value: cyan")
class CyanTriangle(Filled, Polygon):
	color = 'cyan'
	sides = 2
```

# 4. Attribute Magic Methods

## 4-1. @property @setter for getter setters

@property and @attribute.setter magic methods can be used to create a convenient method access interface

```python
class Rectangle:
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self._calculate_area()

	def _calculate_area(self):
		self.area = self.width * self.height

	@property
	def width(self):
		return self.width

	@width.setter
	def width(self, width):
		assert width > 0, f"width {width} must be > 0"
		self.width = width
		self._calculate_area()

	@property
	def height(self):
		return self.height

	@height.setter
	def height(self, height):
		assert height > 0, f"height {height} must be > 0"
		self.height = height
		self._calculate_area()
```

- @property can only be reused within the class chain - cannot be reused among unrelated classes - **use descriptors**

## 4-2. class decorators

We can use a function that returns a class as a decorator

```python
def my_class_decorator(klass):
	klass.name = 'default'
	return klass

@my_class_decorator
class MyClass:
	pass

my_class = MyClass()
print(my_class.name)

```

This can be used for simplifying meta programming features like log tracing.
