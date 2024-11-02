---
title: "🐳 CSS"
outline: deep

tags: "frontend"
updateTime: "2024-10-31 09:38"
---
![云边小卖部](/public/room.jpg)
>为什么天空那么高，因为那些云都是天空的翅膀。

# 初步认识CSS

## 1. 规则
选择器及一条或多条声明  
引入样式时，权重排行：  
`!important > 行内样式 > 内部样式 > 外部样式`  
类比选择器权重排行：  
`!important（放到样式的后面） > 行内样式 > id选择器 > class选择器 > 标签选择器`  
注意：对于包含关系的选择符，要看权重之和，谁权重高听谁话。

## 2. 代码书写规范
- 尽量用展开型而非紧凑型，为了看起来更直观。
- 空格要规范：
  1. 冒号后面要空格。
  2. 选择器和大括号之间要空格。

## 3. 选择器
- 通配符选择器的作用主要是去除所有元素的内外边距。
- 标签选择器：即将有该标签的所有元素都变色。
- 类选择器：定义`.名称`，如`class="名称"`，注意该名称不可使用`p`、`div`等原本就用到的字符，且不能是中文。
- id选择器：`id=""`，与类选择器不同，id只能有一个名称，而类选择器可以有两个命名。
- 群组选择器：通过逗号同时选择几个标签。
- 后代选择器：通过空格选择某个标签的后代，拓展子代选择器（亲儿子选择器）用大于号表示。
- 伪类选择器：用于链接的选择器，初始状态，访问后状态，鼠标悬停状态，点击激活后状态，分别对应：  
  `a:link -> a:visited -> a:hover -> a:active`  
  记忆方法：love -> hate。

### CSS3新增选择器
- `~`：兄弟选择器，如`div ~ li`表示div后面的所有小li。
- `+`：如`div + li`表示div和后面第一个li。
- `[]`：属性选择器，如`[class]`，表示有class就选择，类似还有id、type等。
- `::`：结构伪类选择器，如`li:first-child`表示第一个li，在用于对最后一个或第一个进行操作时很方便。
- `:target`：目标伪类选择器，如`div:target`。

## 文本属性
- `font-size`：字体大小。
- `font-style`：倾斜，`italic normal`，倾斜与正常。
- `font-family`：字体。
- `font-weight`：加粗：100、400、700、900分别对应细、正常、加粗、更粗。
- `text-align`：水平方向对齐方式：`left`、`right`、`center`、`justify`（只对多行起排齐作用，单行文本无作用）。
- `color`：字体颜色。
- `line-height`：行高，通过设置行高和盒子高度相同可以达到单行文本垂直居中对齐的目的，多行无用。
- `letter-spacing`：字符间距，对于中文是汉字之间的距离，对于英文是两个字母之间的距离。
- `word-spacing`：两个英文单词之间的距离。
- `text-indent`：首行缩进。
- `text-decoration`：文本修饰 `underline`、`overline`、`line-through`分别表示下/上划线和删除线，`none`表示没有任何修饰，该用法常在超文本链接中去除下划线。
- `text-transform`：`capitalize`、`lowercase`、`uppercase`分别表示首字母大写、全部小写、全部大写。
- `font`：简写，顺序为倾斜、加粗、大小/行高、字体。

### CSS3新增
- `text-shadow`：文本阴影，如`text-shadow: 10px 10px 1px red;`表示水平位移、垂直位移、模糊程度、阴影部分颜色，可以同时设置两个不同颜色。
- `box-shadow`：盒子阴影。
- `border-radius`：边框圆弧。

## 背景属性
- `background-color`：背景颜色，四种表示方法，直接写颜色、`rgb`、`rgba`（相较于`rgb`多了一个透明度属性）、16进制写法。
- `background-image`：背景图片，`url()`。
- `background-position`：背景图片位置，可以直接用像素大小表示距离左边和上边的距离，也可以用单词表示：水平方向（`left`、`center`、`right`），垂直方向（`top`、`center`、`bottom`）。
- `background-repeat`：背景图片平铺，分为`repeat`（默认平铺）、`repeat-x`（x轴平铺）、`repeat-y`（y轴平铺）、`no-repeat`（不平铺）。
- `background-size`：设置背景图片的大小，可以直接设置长宽，也可以用`cover`（确保将图片整个覆盖盒子）或`contain`（长宽谁如果已经覆盖完就停止覆盖）。
- `background-attachment`：背景图片固定，`scroll`（滚动）、`fixed`（固定）。

### CSS3新增
- 颜色渐变：
  1. 线性渐变：`linear-gradient(,,, )`，支持多颜色渐变，支持方向（`to left`、`to top right`），支持角度写法（`90deg`）。
  2. 径向渐变：波纹状散开。

## 过渡与动画
- `transform` 2D属性：
  - `translate`：平移，默认向右和向下为正值，可以单独设置`translateX/Y`。
  - `scale`：缩放，`scale(x,y)`表示分别沿x、y的放大倍数。
  - `rotate`：旋转，只有一个角度值，顺时针旋转。
- 关键帧动画：
  - 用`@keyframes`自定义名称 `{}` 表示动画效果，表现形式有两种：
    1. `from{} to{}` 从原本的状态过渡到最后的状态。
    2. 百分比 `0% 25% 50% 75% 100%`，可以自由调节。

### 动画过渡类型
- `animation-timing-function`：可调贝塞尔曲线。
- 动画循环次数：数字/infinite（无限）。
- 动画的正反。

## 布局与浮动
- 浮动：`float: left right none` 分别表示左浮动、右浮动、不浮动。
- 清除浮动的方法：
  1. 给浮动元素指定固定高度。
  2. 使用`clear`属性。
  3. 加一个不指定宽高的盒子对其清除浮动。
  4. 父元素设置`overflow: hidden`。
  5. 最完美的方法：父元素设置伪元素 `::after` 进行清除。

### 盒子模型
- 理解内外边距：
  1. 边框：三要素：宽度、样式、颜色。
  2. 内边距：`padding`，可以单独设置。
  3. 外边距：`margin`，可以复写。

## 溢出属性
- `overflow`：包括以下几种方式：`visible`、`hidden`、`scroll`、`inherit`、`auto`。
- 空余空间：`white-space`：`normal`、`nowrap`、`pre`、`pre-wrap`、`pre-line`、`inherit`。

### 单行文本溢出部分省略号显示
1. 指定宽度。
2. `white-space: nowrap`。
3. `overflow: hidden`。
4. `text-overflow: ellipsis`。

## 显示属性
- `display` 四种属性：`block`、`inline`、`inline-block`、`none`。
- 定位：`position`，通常子绝父对。
- 透明度：`opacity`。

### 自适应布局
- 窗口自适应时设置`html, body { height: 100%; }`。
- 实践操作：两栏布局（两种方法实现）。
## flex布局

**flex**：单词弹性的缩写（flexible）

## 与传统布局的区别

- 在移动端中使用更方便
- pc端一些均匀分布或者居中分布时使用更方便

## 使用原理

给某元素添加`display：flex` 后，该元素相当于一个容器，所有**子元素**标签都可以作为容器被赋予宽高

### 给父元素添加的属性

`flex-direction`:主轴方向，默认为row行，可设置为colunm列

`justify-content`:主轴上标签排列方式，默认为`flex-start`即从左向右排列

还可以设置：

flex-end  :从后向前排列

space-between ：先两边贴库，中间剩余空间平分

space-round ：不贴库，直接平分

center：居中显示

```css
			display: flex;
            flex-direction: column;
            flex-wrap: wrap;//设置是否换行，默认不换行
            justify-content:space-round;
            align-items: flex-start;
            align-content: ;
            flex-flow: column wrap;
```

`align-items`：只有单行时可用，用来调整侧轴的方式，可用属性有：

flex-start

flex-end

center

**stretch**:拉伸，将子元素在该方向拉伸至父元素同样长度

`align-content`:只有多行时能用，用来调整侧轴排列方式,可用属性有：

flex-start

flex-end

center

**stretch**:拉伸，将子元素在该方向拉伸至父元素同样长度

space-between ：先两边贴库，中间剩余空间平分

space-round ：不贴库，直接平分

`flex-flow`:复合属性，主轴方向和是否换行的复合体

*******

### 给子元素添加的属性

**flex**：相当于一个权重，用来分剩余空间

例如：

```css
div {
 display:flex;
}
div span{
	flex:1;
}
```

如果有三个`span`，则这三个span平分剩余空间

`align-self`:给某个子元素设置后，可脱离父元素align-item的控制，自己单独设置align-item属性值
## 其他注意事项
- 使用`calc`函数时，注意减号前后要有空格。
- 图片整合后通过`background-position`可以节省访问次数，加快访问速度（雪碧图）。
- 防止空内容不好看，可以设置`min-height`、`min-width`。

--- 
