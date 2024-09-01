# <Typescript> JSON.stringify Can't Serialize Map Type
While working on FFXIV Simbot, I found that by equipment database JSON string is empty:

ex) 
```javascript
let equipmentSlotNameToIdMap: Map<string, number> = new Map();
equipmentSlotNameToIdMap.set("weapon", 10342);

JSON.stringify(equipmentSlotNameToIdMap); // gives "{}"
```

* A very bad bug that took me hours to figure out. Turns out that **Javascript JSON.stringify does not support Map types**
* To solve it, we need to change the Map type to something Serializable, like an array.


```javascript
const WEAPON_EQUIPMENT_SLOT_ID = 0;
let equipmentSlots: number[] = [];

equipmentSlots[WEAPON_EQUIPMENT_SLOT_ID] = 10342;

JSON.stringify(equipmentSlots); // gives "[10342]"
```