# 3. Using Flexbox
* Flexbox itself is block - uses maximum width
* All direct childs of a flexbox is flex

## axis
flexbox always has direction - default horizontal(left to right)
which means **main axis goes from left to right(along the row)**

cross axis goes top to bottom

```css
.container {
  border: 5px solid #ffcc5c;
  display: flex;
  flex-direction: row; /* default */
  flex-direction: column; /* stacks from top to bottom - main axis is now top to bottom */
}
```

## justify
* justify items along main axis(default left to right) - **justify-content: controls the content along main axis**

```css
.container {
  border: 5px solid #ffcc5c;
  display: flex;
  justify-content: flex-start; /* default: squeezed from left to right */
  justify-content: flex-end; /* squeezed to right side */
  justify-content: center;
  justify-content: space-around; /* equal amount of space to left and right  ex) |  a    b    c  | */ 
  justify-content: space-between; /* equal amount of space to left and right remove left end and right end margin ex) |a    b    c|*/
  justify-content: space-evenly; /* equal amount of space for every component and margin ex) |  a  b  c  | */
}
```


## positioning
```css
.logout {
    margin-left: auto; /* moves component to right end */
    margin-right: auto; /* moves component to left end */
    margin: auto; /* center */
}
```

## flex property
* components size grow based on parent size
* flex: shorthand for
   1) flex-grow
   2) flex-shrink
   3) flex-basis
```css
.container > div {
  flex: 1; /* better than setting size to percentage like width: 33% b/c you have to change number every time you add new container */
  flex: 2; /* twice the width of flex: 1 elements */
}
```

very concise, but not really used in many cases -> usually **some components are fixed and some are flexed - remove flex: 1 from fixed components**


## align items 
* control items in "cross axis" - by default they "stretch themselves" across cross axis - maximum value possible

```css
.container {
  border: 5px solid #ffcc5c;
  display: flex;
  align-items: stretch; /* default */
  align-items: flex-start; /* start of cross axis, and take up only the space they need */
  align-items: flex-end; /* end of cross axis, and take up only the space they need */
  align-items: center; /* center of cross axis */
}

.logout {
  align-self: flex-start; /* only justifies this one component to start */
}
```

sidenote - **flexbox is great for centering an item inside container**


## wrapping
* setting px width will set **maximum value - if the width can't be done cause the container is too short, it won'd become that value**

```css
.container {
  border: 5px solid #ffcc5c;
  display: flex;
  width: 600px;
}

/* there are three divs in container */
.container > div {
  width: 300px; /* this is maximum possible value - right now there are three divs so width becomes 200px each */
  /* if container becomes > 900px, then divs become 300px */
}
```

this is because **flexbox has flex-wrap property set to nowrap as default - only one row/column along your main axis**
this can be changed by setting flex-wrap: wrap

```css
.container {
  border: 5px solid #ffcc5c;
  display: flex;
  flex-wrap: wrap;
  /* if width < 600px, all divs in separate rows */
  /* if width < 900px, first two divs in one row and last one in second row */
  /* if width >= 900px, all divs in one row */
}
```

## flex grow/shrink/basis
```css
.home {
  flex: 1;
}

/* is shorthand for */
.home {
    flex: 1 1 0;
}

/* which is also shorthand for */
.home {
    flex-grow: 1;
    flex-shrink: 1;
    flex-basis: 0;
}

.home {
    flex-basis: 200px; /* they will be max 200px even if container is longer, but they will shrink when container is too short */
    flex-grow: 1; /* decides the ratio of extra space should be distributed to various items when container is long - if 0, will not be distributed - values don't matter: ratio between elements are more important - flexgrow 1:1 = 5:5 */
    flex-shrink: 1 /* decides the ratio of shrinking space when container is short */
}
```


## order
* Flexbox can "sort order independance" - we can move out items regardless of HTML order
* by default **order is set to 0**

```css
.item2 {
  order: -1; /* goes to first, since default is 0 */
  order: 1; /* goes to last, since default is 0 */
}
```