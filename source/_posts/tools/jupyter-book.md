---
title: jupyter-book
date: 2021-08-24 10:40:39
tags:
---
介绍使用 jupyter-book 写博客（通常是一个在线电子书）的方法。
<!-- more -->

Jupyter Book [官方文档](https://jupyterbook.org/start/overview.html)

另一个文档在[这里](https://predictablynoisy.com/jupyter-book/guide/01_overview)。

使用 Jupyter Book 可以创建基于 markdown 的文档，并使用 `jupyter-book` 将这些文档转为托管到 web（例如 github）的电子书。

安装步骤：
```sh
conda create -n jupyter python=3.9 # 新建一个专用 env
conda activate jupyter
pip install jupyter-book
```

# 创建
创建一个 Jupyter Book，
```sh
jupyter-book create mybookname/
```

# 文件目录结构
## 内容文件
有两种文件格式 `.md` 和 `.ipynb`，前者是标记文档，主要包含文本内容，后者则包含了计算内容（代码）和叙述内容（文本）。这两者不多说，大家或多或少都了解。

## 配置文件
`_config.yml`

## 内容表
 Table of Content：`_toc.yml`


# 生成
如果当前目录在 `mybookname` 父级目录，那么执行
```sh
jupyter-book build mybookname/
```
生成的文件全部位于 `mybookname/_build` 文件夹下。

如果当前目录在 `mybookname`，那么执行
```sh
jupyter-book build .
```
因为有 cache，未修改的 source 文件不再重新生成，如果需要全部重新生成，执行
```sh
jupyter-book build --all .
```

如果报错：
```
ImportError: DLL load failed while importing win32api: The specified module could not be found
```
解决方法：
```sh
python C:\path\to\miniconda3\Scripts\pywin32_postinstall.py -install
```

所有的内容文件和配置文件均为 source 文件，生成的文件位于 `_build` 目录，称为 build 文件。 在 git 项目中，main 主分支中 ignore `_build` 目录，并将 `_build` 中文件 push 到 另一个分支（例如 `gh-pages`）中，用于在 web 中展示。

# 预览
可以在本地文件`_build/html/index.html` 上双击打开，也可以在浏览器中输入文件路径 （例如 `file://Users/my_path_to_book/_build/html/index.html`）

# 发布上线

## 创建在线仓库
在 github 上创建新仓库，仓库名可取为 `mybook`，可为仓库添加一句描述，仓库初始化时不要 `README` 文件，即完全空仓库，然后 clone 仓库到本地，
```sh
git clone https://github.com/<my-account>/mybook
```

将 book 中的所有文件和文件夹拷贝到这个本地仓库，
```sh
cp -r mybookname/* mybook/
```

添加 `.gitignore` 文件，文件内容为，
```
mybook/_build/*
.ipynb_checkpoints
.DS_Store
__pycache__/
```

然后同步到本地仓库和远程仓库，
```sh
cd mybook
git add ./*
git commit -m "adding my first book"
git push
```

## 使用 Github Page 将 book 发布上线
我们已经将源文件推送到 github 库，此时还需要将生成的文件发布上线，使得生成一个 web 页面。最简单的办法是使用 `ghp-import` 包，

`ghp-import` 工作机制是将生成的内容 （`_build/html` 目录）拷贝到仓库分支 `gh-pages`，并推送到 github，`ghp-import` 自动创建`gh-pages` 分支并填充生成的文件。

操作步骤：

1. 安装
    ```sh
    pip install ghp-import
    ```

2. 更新 Github pages 站点设置
    - 使用 `gh-pages`分支托管网站

3. 在 `main` 分支下（即 `master` 分支，不包含 `_build/html` 文件夹），调用
    ```sh
    ghp-import -n -p -f _build/html
    ```
    `-n` 选项是告诉 Github 不用使用 `Jekyll` 来生成 book，因为我们的 HTML 已经生成

几分钟之后，book 可以通过 `https://<user>.github.io/mybook/` 访问。

## 修改 book
checkout 仓库的 `main` 分支，然后进行修改 book 内容，然后 re-build `jupyter-book build .`，然后使用 `ghp-import -n -p -f _build/html` 推送到 `gh-pages`。