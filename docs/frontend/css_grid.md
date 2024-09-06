# Grids

## Fraction units and repeat
```css
.container {
    display: grid;
    /* three items in one row: the first and last item is 100px and the center one responsively changes */
    grid-template-columns: 100px auto 100px;
    /* two rows, 50px each */
    grid-template-rows: 50px 50px;
    grid-template-columns: 1fr 2fr 1fr; /* three items in one row, the center one being 2 times the length of the first and last one */
    grid-template-columns: repeat(3, 1fr); /* equal to 1fr 1fr 1fr */

    grid-template: /* row */ repeat(2, 50px) / /* column */ repeat(3, 50px);
    grid-gap: 3px;
}

```


## Positioning Items
```css
.container {
    display: grid;
    grid-gap: 3px;
    grid-template-columns: repeat(2, 1fr);
    grid-template-rows: 40px 200px 40px;
}

.header {
    grid-column-start: 1; /* header starts at first column line (= leftmost) */
    grid-column-end: 3; /* and ends at third column (= rightmost) */
    /* there is three column lines b/c the margin lines are also counted as a column line
       ex) | col1 | col2 |:
           |(first column line) col1 |(second column line) col2 |(third column line) */

    /* these are all equal */
    grid-column: 1 /* start */ / 3 /* end */;
    grid-column: 1 /* start */ / span 2;
    grid-column: 1 / -1 /* the last column line */
}
```

## Template Areas
instead of indexing areas in grid, we can assign them string names
```css
.container {
    height: 100%;
    display: grid;
    grid-gap: 3px;
    grid-template-columns: repeat(12, 1fr);
    grid-template-rows: 40px auto 40px;
        grid-template-areas: 
        ". h h h h h h h h h h ." /* . excludes that area */ 
        "m c c c c c c c c c c c"
        ". f f f f f f f f f f .";
    grid-template-areas: 
        "h h h h h h h h h h h h" /* 12 * 3, same as the size of grid */
        "m c c c c c c c c c c c"
        "f f f f f f f f f f f f";
}
.header {
    grid-area: h;
}

.menu {
    grid-area: m;
}

.content {
    grid-area: c;    
}

.footer {
    grid-area: f;
}
```

## Auto-fit and Minmax
```css
.container {
    display: grid;
    grid-gap: 5px;
    grid-template-columns: repeat(auto-fit, 100px) /* fits as many columns as possible in the current width, which will be floor(width / 100) for this case */;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); /* if some frames are left as remainder when setting each column to 100px, use 1fr instead - always uses full width */
    grid-template-rows: repeat(2, 100px);

    /* 
      Since number of rows change depending on width(since width is auto-fit), we can't explicitly state the number of rows, so we use auto-rows
    */
    grid-auto-rows: 100px; 
}
```


## Named Lines
```css
.container {
    height: 100%; 
    display: grid;
    grid-gap: 3px;

    /* blanks are column spaces */
    /*main-start content-start     content-end|main-end*/
    grid-template-columns: [main-start] 1fr [content-start] 5fr [content-end main-end];
    /* 
       main-start
        (40px)
       content-start
        (auto)
       content-end
        (40px)
       main-end
    */
    grid-template-rows: [main-start] 40px [content-start] auto [content-end] 40px [main-end]; 
}

.header {
    grid-column: main; /* adds postfix -start, -end */ 
}

.menu {}

.content {
    grid-area: content;
}

.footer {
    grid-area: main; /* this does not work cause header and menu content already took up the main-start to content-end rows */
    grid-column: main; /* can only use columns instead */
}

```


## justify and align
```css
.container {
    border: 1px solid black;
    height: 100%;
    display: grid;
    grid-gap: 5px;
    grid-template-columns: repeat(3, 100px);
    grid-template-rows: repeat(2, 100px);

    /* justify: columns */
    justify-content: start; /* moves all items to left side */
    justify-content: center; /* moves all items to center side */
    justify-content: end; /* moves all items to right side */
    justify-content: space-between; /* much space as possible along column */

    /* align: rows */
    align-content: start; /* moves all items to up side */
    align-content: center; /* moves all items to center row */
    align-content: end; /* moves all items to down side */

    space-evenly; /* much space as possible along column, no space at edges */
    space-around; /* much space as possible along column, half space at edges */

    /* items: stretch by default */
    justify-items: stretch | center | start | end;
    align-items: stretch | center | start | end;
}

.content {
    grid-column: 3 / -1;
    justify-self: center; /* only justifies itself */
    align-self: end; /* only aligns itself */

}
```
