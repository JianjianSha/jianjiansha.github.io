---
title: Hexo Sync
date: 2019-06-13 9:57:11
tag: tool
---

场景：
```
在A, B两台电脑上同步Hexo博客
```
假设在 computer B 上已经初次建立hexo博客 https://shajian.github.io， computer B 本地的文件夹（hexo部署环境目录）为 path/to/myblog，其内部文件/目录如下：
```
_config.yml
db.json
node_modules
package.json
package-lock.json
public
scaffolds
source
themes
```
在github仓库 shajian.github.io 上新建branch，比如"hexo"，这样，"mater"主分支用于维护hexo生成的静态博客文件/目录，"hexo"分支用于维护hexo部署环境下的所有文件/目录。

在 computer A 上 clone 这个仓库，并切换到 hexo 分支，
```
$ git clone https://github/shajian/shajian.github.io.git
$ cd shajian.github.io
$ git checkout hexo
$ git branch
*hexo
 master
```
将目录 shajian.github.io 内的所有文件/目录全部删除，然后将 path/to/myblog内的全部内容复制过来，
```
$ rm -rf .
$ cp path/to/myblog/* ./
```
然后可在使用
```
hexo init "<title>"
```
写新文章或直接去source/_posts下修改已有文章，
部署
```
hexo g -d
```
然后提交到仓库的hexo分支，进行备份
```
$ git add .
$ git commit -m "new post 'title'"
$ git push origin hexo
```

然后就可以去 https://shajian.github.io 浏览本地新增/修改文章内容了。

在 computer B 上删除 path/to/myblog 目录，然后重新 clone 仓库，并切换到 hexo 分支，
```
$ git clone https://github/shajian/shajian.github.io.git
$ cd shajian.github.io
$ git checkout hexo
```
如果仓库有 .gitignore 文件且包含 node_modules 目录，则执行
```
$ npm install
```
此时，要修改还是新增文章，步骤均与上面 computer A上的操作一致。

computer A 和 B 本地均有仓库后，以后每次修改还是新增文章，首先需要将仓库更新到最新
```
$ git checkout master
$ git pull origin master
$ git checkout hexo
$ git pull origin hexo
```
切换到 hexo 分支后，可以进行修改和新增文章了。