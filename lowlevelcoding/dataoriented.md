# 2. Data Oriented Programming
Object Oriented Programming is the current norm, but the downsides of OOP is being noticed by the programming community

Benefits of OOP: **Simplifies complex problems by dividing them into smaller components**
Downsides of OOP: **Solved problem in an "ideal world" - overprogramming, oversimplification**

OOP 'encapsulating' data into classes and limiting the allowed use of the class by methods **helps robustness; people without thorough knowledge of the program won't be able to break the system by doing some unexpected activity.**
   * all methods are robustly tested for edge cases and guaranteed to work well - we can control the allowed actions so that it doesn't break

However, for the cost of this robustness, **usually performance is sacrificed**
   * you can only do actions in "limited choices," but usally the "most performance optimal ways" are specific in every situation
   * however, we can only use the general methods for reusability, sacrificing performance for ideal simplicity

To sum it up, **current OOP is way too focused on human readability - we need to also consider machine-friendlyness to be a better programmer.**


## 2-1. No useless branching - separate all cases
* Having booleans in struct/classes have two downsides:
   1) **sparse information**: boolean is a 1-bit information packed in a byte(at least 7 bit wasted)
   2) **branching**: having a bool means you're going to branch with that boolean, resulting in additional branching

To solve this, just **separate structs depending on branch**

```cpp
// 12 byte
struct Person {
    id: int
    has_drivers_license: bool
    drivers_license_id: int
}

Person p;

// Case is divided to person with driver's license and person without. Divide them so that we don't have to branch!!!!
int get_drivers_license_id(Person p) {
  if p.has_drivers_license {
    p.drivers_license_id
  } else {
    -1
  }
}


// Optimization - 8 byte
struct PersonWithDriversLicense {
    id: int,
    drivers_license_id: int
}

// 4 byte
struct PersonWithoutDriversLicense {
    id: int
}

PersonWithDriversLicense p1;

p1.id
```