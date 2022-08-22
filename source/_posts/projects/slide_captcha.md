---
title: 滑动验证码破解
date: 2022-05-06 10:41:10
tags: captcha
---

滑动验证码网站：https://captcha1.scrape.center/

# 1. 安装油猴插件

以 chrome 为例，

1. 打开 chrome，输入 `chrome://extensions` 打开浏览器扩展页面
2. 点击左侧三横线，然后选择下方 `open Chrome Web Store`，打开 Web 应用商店
3. 搜索栏输入 `Tampermonkey`，安装油猴插件

点击浏览器中油猴插件，`Find new scripts` 可以安装在线脚本，`Create a new script` 可以使用自己编写的脚本，这里我们点击 `Create a new script`，然后脚本中输入 
```js
// ==UserScript==
// @name         hook createElement
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       朱宇
// @match        *://*/*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    // Your code here...
    let _createElement = document.createElement.bind(document);

    document.createElement = function (elm) {
        console.log("createElement:",elm);
        if (elm === "canvas") {
            debugger;
        }
        return _createElement(elm);
    }
})();
```

我们即将使用这个脚本进行调试。

# 2. 破解滑动验证码

先将上述自定义脚本中的 `debugger` 一行注释，然后

打开网站 https://captcha1.scrape.center/，按 F12，打开开发者窗口并切换到 `Network`，然后点击页面上 `登录` 按钮，弹出验证码（如果没有弹出，可以尝试多次点击）。也可以点击验证码窗口上的 `刷新` 按钮，刷新验证码。

通过观察 `Network` 请求列表发现，每一个验证码的出现，伴随两个或者三个图片请求，如图 1，

![](/images/projects/slide_captcha1.png)

<center>图 1. </center>

猜测是跟所用的请求 url 有关，refresh 请求有一组 url 可供选用，有的 refresh url 对应两个图片请求，有的 refresh url 对应三个图片请求，如图 1 红色框部分，

1. `a67f7d022.webp`：这是一个（打乱 patch 顺序）验证码图片（有缺口阴影）
2. `a67f7d022.png`：这是对应的滑块图片
3. `2fccf95e6.webp`：（可选，有的 refresh url 没有这个请求），这是一个滑块放对位置后（即没有缺口阴影）的验证码图片，patch 顺序同样打乱。

refresh url 请求

`refresh.php?gt=afb55317868a27591e984c714e11f512&ch...&callback=geetest_1651803980633` 的返回结果为，
```json
geetest_1651803980633({challenge: "38e8220880e540762921f9c6726a057a6k", id: "", type: "multilink",…})
bg: "pictures/gt/2fccf95e6/bg/a67f7d022.jpg"
challenge: "38e8220880e540762921f9c6726a057a6k"
feedback: "http://www.geetest.com/contact#report"
fullbg: "pictures/gt/2fccf95e6/2fccf95e6.jpg"
height: 160
id: ""
link: ""
slice: "pictures/gt/2fccf95e6/slice/a67f7d022.png"
type: "multilink"
xpos: 0
ypos: 63
```
返回结果分析：
1. `xpos, ypos` 应该就是滑块图片位于验证码图片上的初始位置
2. `slice` 滑块图片的 url，后续将请求这个 url 获得滑块图片
3. `fullbg` 没有缺口阴影的验证码图片 url （打乱 patch）
4. `bg` 有缺口阴影的验证码图片 url （打乱 patch）
5. `height` 验证码图片高度

## 2.1 获取验证码图片

前面拿到的验证码图片 patch 顺序被打乱，如何恢复？这就用到油猴以及我们给出的自定义脚本。

__自定义脚本代码分析__：

右键验证码查询元素发现这是一个 canvas，而右键页面选择 `view page source` 查看页面 html 源码发现并没有 canvas，猜测这个 canvas 由 js 生成，然后再使用 canvas 的自带方法设置（由 js 恢复正确 patch 顺序后的）图片。于是我们自定义 js 脚本， hook 住每个 element 的创建，如果发现 element 是 canvas，那么进入断点调试。

将上述自定义脚本中的 `debugger` 一行取消注释，然后点击`刷新`按钮，在 `debugger` 一行断点，然后单步调试，跳出我们自定义脚本中的函数之后，回到 `$_BDM(t,e)` 这个函数中，如图 2，

![](/images/projects/slide_captcha2.png)

<center>图 2. </center>

单步调试（F10）到 291 行，在控制台执行 

![](/images/projects/slide_captcha3.png)

可以得到恢复 patch 顺序之后的图片，这是一个 Data URI scheme，直接粘贴引号内的字符串，可以打开这个图片，

![](/images/projects/slide_captcha4.png)

发现这是一个没有缺口阴影的图片，所以猜想还要进行一次 patch 顺序恢复操作，以便获得 __有缺口阴影的验证码图片__，在 291 行增加一个断点，然后直接点击运行（F8，或者鼠标点击蓝色三角形按钮），跳至 `debugger` 一行，继续运行，跳至 291 行暂停，此时再在控制台执行 `s.canvas.toDataURL()`，得到有缺口阴影的验证码图片。

# 3. another 验证码

顶象：https://user.dingxiang-inc.com/user/bind#/

打开页面，输入手机号，点击 `下一步` 按钮，弹出验证码窗口，右键查看元素，也是一个 `canvas` 标签。调试发现，并没有命中我们自定义脚本的调试断点。于是选择 `Sources` （位于 F12 窗口中），在左侧树形结构 `top` 顶级节点搜索 `canvas` 关键词，找到几处包含 `canvas` 的 js 代码，分别打上断点，进行调试，调试发现处理 patch 乱序的图片 的代码如图 5，

![](/images/)

<center>图 5.</center>

图 5 中，执行到 6528 行，就可以得到校正后的图片了，在控制台执行 `l.toDataURL()` 获得图片的 base64 字符串。图片的恢复操作（纠正 patch 顺序）应该是图 5 中红色框部分的代码：
1. `b`： 一个函数，依次根据 `s` 中的 patch 顺序进行重绘，即调用 `b` 的第二个参数
2. `s`：index 数组，表示patch 打乱后的顺序

`b` 的第二个参数是一个函数，`function(n, i) {...}`，其中
1. `n`： `s` 数组中的当前元素值，表示 patch index
2. `i`： 循环 index，从 0 到 `s.length`
3. `d`： CanvasRenderingContext2D 对象，用于绘图，参考方法

    `void CanvasRenderingContext2D.drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)`

# 4. 数据抓取时破解流程

大概思路：一个验证码，唯一地对应一组参数（例如 `gt` 和 `chanllenge`），数据抓取时，例如发送查询请求，返回的响应中检测到有验证码弹出，此时应使用 params 然后请求一个专门的 web 接口以获得相同的验证码。

1. 打开首页 https://captcha1.scrape.center/

    通过抓包，发现有如图 6 中的蓝色背景的请求，响应数据包含了 `gt` 和 `challenge`。

    ![](/images/projects/slide_captcha6.png)
    <center>图 6 </center>

2. 点击登录按钮，此时弹出验证码窗口，抓包发现新的请求，如图 7，

    ![](/images/projects/slide_captcha7.png)
    <center>图 7 </center>

    重点关注红色框那个请求，请求参数 `gt` 和 `challenge` 的值正好是前面返回的值。这个请求返回的数据包含了：

    - `gt`：下一个验证码对应的 gt 值。这里这个参数实际上自从打开首页就保持不变。
    - `challenge`：下一个验证码对应的 challenge。
    - `bg`：验证码背景图片（带缺口）
    - `slice`：滑块图片
    - `fullbg`：完整的验证码图片（不带缺口）
    - `xpos`, `ypos`，滑块图片在背景图片上的初始位置

    我们首次请求网站首页，得到 `gt` 和 `challenge` ，发送数据请求后，如果返回的响应中包含了验证码，那么可以发送这个 `get.php?is_next=true&type=slide3&gt=...` 得到验证码图片地址，__同时更新 `gt` 和 `challenge`__（这里当然也可以不用更新 gt）。

    当然，在页面上我们还可以点击刷新按钮，以得到验证码图片，如图 8， 

    ![](/images/projects/slide_captcha8.png)
    <center>图 8 </center>

    注意观察，这个请求中 `challenge` 参数值已经更新了，同时返回的数据中也包含了 `challenge`，`fullbg`，`bg`，`slice`，`xpos`，`ypos` 等。

    由于这里是登录功能的验证码，与数据抓取中的验证码在处理逻辑上不同，故以上分析逻辑不适用于数据抓取，不能完全照搬，仅供参考。



# 5. 验证码图片抓取

上面分析 js 代码，目的是为了分析图片恢复的逻辑，以便使用 python 实现同样的逻辑，从而避开 js。

使用 python+selenium 进行验证码图片抓取。

selenium 点击`登录`或者 `刷新` 按钮之后，寻找 canvas 元素，注意 class 属性是 `geetest_canvas_bg`，

```js
// 这里使用 js 代码，仅作过程演示，
// 实际抓取时，使用 selenium api 进行元素定位
canvas = document.getElementsByClassName("geetest_canvas_bg")[0]
// 获取图片 base64 编码
img = canvas.toDataURL()
```

参考：
1. https://www.cnblogs.com/zhuchunyu/p/12455804.html