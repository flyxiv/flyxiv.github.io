# 1. FFXIV Stat Theory(7.0 Dawntrail)

## 1-1. Main Stat Base Per Job
### a. Base Main Stats By Job
Every job has a base main stat amount which decides the initial main stat for the class at each level.
   * these bases are shown in https://www.akhmorning.com/allagan-studies/modifiers/
   * in Dawntrail, the initial main stat at lvl 100 is calculated by **stat base * 4.4**

> ex) Warrior's STR base is 105 -> initial STR of Warrior @ lvl 100 = 105 * 4.4 = 442

### b. Racial Stats
* Racial stats are well known and organized in many FFXIV sites. The highest value is 23 and lowest is 18, so there is a 5 point maximum difference between each race.


By adding the two factors we can calculate **The initial main stat for each job, for each race**.

> ex) Initial STR of a Warrior, Hyur Highlander: 442 + 23 = 465


## 1-2. Zero-base values of stats
* For Dawntrail, the **zero-base values of all substats except Piety and Determination is 420**
* For **main stats and piety/determination, the zero-base value is 440.**


## 2-1. Calculating Main Stat Multiplier
* The increase slope of the main stat multiplier **is different for tanks and non-tanks**
* The basic formula is as below:
> Main Stat Multiplier = (Floor((slope)(Main Stat Value - 440)) + 100)%
> slope: 0.4 for Tanks, 0.5 for Non-Tanks(Healer, DPS)

> ex) if an AST's total Main Stat is 1000, then its Main Stat Multiplier = FLOOR(560*0.5) + 100% = 380%
> 380% -> a skill of 100 potency will be increased to 380 

## 2-2. Calculating Sub Stat Multiplier
First we define "Effective Stat Increase", which will make the equations way easier to read.
> Effective Stat Increase(Stat) = Stat - Stat Base Value
> ex1) If Critical Strike = 1000, Effective Stat Increase = 1000 - 420 = 580
> ex2) If Determination = 1000, Effective Stat Increase = 1000 - 440 = 560

### a. Critical Hit Rate
>>> Critical Hit Increase = FLOOR(200*(Effective Stat Increase)/2780)/10% = FLOOR(Effective Stat Increase / 13.9) / 10 = 0.1% increase every 13.9 Stat
>>> Critical Strike Rate = 5% + Critical Hit Increase
>>> Critical Strike Damage = 140% + Critical Hit Increase

### b. Direct Hit Rate
>>> Direct Hit Rate = FLOOR(550*(Effective Stat Increase)/2780)/10 % = FLOOR(Effective Stat Increase * (55 / 278)) / 10 = 0.1% increase every 5.05454... Stat
### c. Determination 
>>> Determination Increase = FLOOR(140*(Effective Stat Increase)/2780)/10% = FLOOR(Effective Stat Increase * (14 / 278)) / 10% = 0.1% increase every 19.857 Stat
>>> Determination Damage Bonus = 100% + Determination Increase

### d. Tenacity
>>> Tenacity Damage Increase = FLOOR(112*(Effective Stat Increase)/2780)/10% = FLOOR(Effective Stat Increase * (112 / 2780)) / 10% = 0.1% increase every 24.82 Stat
>>> Tenacity Damage Reduction Increase = FLOOR(200*(Effective Stat Increase)/2780)/10% = FLOOR(Effective Stat Increase / 13.9) / 10% = 0.1% increase every 13.9 Stat
>>> Tenacity Damage Bonus = 100% + Tenacity Damage Increase
>>> Tenacity Damage Reduction = 100% - Tenacity Damage Reduction (if it is 98%, a 100 damage becomes 98)

### e. Piety
>>> Piety Increase = FLOOR(150*(Effective Stat Increase)/2780) = FLOOR(Effective Stat Increase * (150 / 2780)) = 1MP Tick increase every 18.53 Stat
>>> Mana Increase Per Tick = 200 + Piety Increase

### f. Speed(Spell/Skill)
>>> Speed Increase = FLOOR(130*(Effective Stat Increase)/2780) * 10% = FLOOR(Effective Stat Increase * (13 / 278)) / 10% = 0.1% increase every 21.38 Stat
>>> Auto/DOT Attack Increase = 100 + Speed Increase %
>>> GCD Speed(s) = FLOOR(GCD(ms) / (100 + Speed Increase) * 100) / 100
>>> ex) if Speed Increase = 110%, then GCD for a 2.5s = FLOOR(2500 / 1.10) / 100 = FLOOR(227.27...) / 100 = 2.27s



