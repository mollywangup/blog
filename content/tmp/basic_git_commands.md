- [安装](#安装)
- [环境配置](#环境配置)
  - [设置本地免密](#设置本地免密)
  - [全局配置](#全局配置)
- [初始化仓库](#初始化仓库)
  - [结构理解](#结构理解)
  - [创建本地仓库](#创建本地仓库)
  - [克隆远程仓库](#克隆远程仓库)
- [常用操作](#常用操作)
  - [git remote](#git-remote)
  - [git push](#git-push)
  - [git pull](#git-pull)
  - [git branch](#git-branch)
- [SSH和HTTPS的主要区别](#ssh和https的主要区别)
- [常用.gitignore配置](#常用gitignore配置)
- [常见报错解决](#常见报错解决)


# 安装
```shell
# 安装
sudo apt-get install git

# 卸载
sudo apt-get remove git

# 验证安装
git --version
```

# 环境配置
## 设置本地免密
Step1: 先检查SSH权限，如果已有SSH权限，则直接结束该步骤；如果没有SSH权限，则需要跳到下一步。方法如下：
```shell
ssh -T git@github.com

# git@github.com: Permission denied (publickey).

# Hi xxx! You've successfully authenticated, but GitHub does not provide shell access.
```

Step2: 获取或直接创建本地SSH公钥，方法如下：
```shell
# 获取方法
cat ~/.ssh/id_rsa.pub

# 创建方法
ssh-keygen
```

Step3: 将本地SSH公钥添加至GitHub后台。方法见 [Error: Permission denied (publickey)](https://docs.github.com/en/authentication/troubleshooting-ssh/error-permission-denied-publickey)


## 全局配置
```shell
# 配置默认分支为main
git config --global init.defaultBranch main

# 配置默认账户
git config --global user.email "mollywangup@gmail.com"
git config --global user.name "mollywangup"

# 配置默认的编辑器VS Code（可选）
git config --global core.editor "code --wait"

# 验证配置
cat ~/.gitconfig 
```

# 初始化仓库
## 结构理解
<img src="https://user-images.githubusercontent.com/46241961/214644219-90d746a9-b0ab-4fd5-9e3d-1bc7bac16ef0.png" title="refs" width=30%>

## 创建本地仓库
```shell
# Step1: 初始化
git init

# Step2: 设置关联远程
git remote add origin git@github.com:username/repository.git

# Step3: 修改默认分支名称
git branch -m main

# Step4: 首次提交
touch README.md
echo "test" >> README.md
git add .
git commit -m "first commit"
git push -u origin main
```


## 克隆远程仓库
```shell
# Step1: git clone
git clone git@github.com:username/repository.git

# Step2: git pull
git pull origin main
```


# 常用操作
## git remote
- 添加关联远程: 
  ```shell
  git remote add origin git@github.com:username/repository.git
  ```
- 移除关联远程: 
  ```shell
  git remote rm origin
  ```
- 修改关联远程: 
  ```shell
  git remote set-url origin git@github.com:username/newrepository.git
  ```
- 查看关联远程: 
  ```shell
  # 方法一
  git remote -v

  # 方法二
  cat .git/config
  ```


## git push
- 查看变更: `git status`
- 添加变更: `git add <file>`
- 提交变更: `git commit -m <message>`
- 提交至远程: `git push origin <branch>`
- 查看提交记录: `git log origin/<branch>`
- 撤销本地提交: 
  ```shell
  # 保留有改动的代码，撤销commit, 保留add
  git reset --soft HEAD^

  # 保留有改动的代码，撤销commit, 撤销add
  git reset --mixed HEAD^

  # 删除有改动的代码，撤销commit, 撤销add, 即恢复到上一次的commit的状态
  git reset --hard HEAD^
  ```

## git pull
- 先拉取变更再合并至工作区: 
  ```shell
  git fetch origin <branch>
  git merge origin/<branch>
  ```
- 直接拉取变更至工作区: 
  ```shell
  # 如果当前分支是主分支
  git checkout main
  git pull --rebase origin

  # 如果当前分支不是主分支，则必须指定一个具体的分支
  git pull --rebase origin <branch>
  ```


## git branch
- 查看分支: `git branch -a`
- 新建分支: `git branch <new-branch>`
- 切换分支: `git checkout <new-branch>`
- 新建并切换分支: `git checkout -b <new-branch>`
- 将本地分支提交至远程: `git push origin <branch>`
- 删除本地分支: `git branch -d <branch>`
- 删除远程分支: `git push origin :<branch>`
- 重命名分支:
  ```shell
  # 方法一：切换到需要重命名的分支操作
  git checkout <old-branch>
  git branch -m <new-branch>

  # 方法二：切换到主分支操作
  git checkout main
  git branch -m <old-branch> <new-branch>
  ```


# SSH和HTTPS的主要区别
```shell
# SSH方式（推荐）
git xxx git@github.com:username/repository.git

# HTTPS方式
git xxx https://github.com/username/repository.git
```

# 常用.gitignore配置
```shell
# 一般以下文件后缀直接忽略
.git
.DS_Store
__pycache__
*.png
*.jpg
*.svg
*.xlsx
*.csv
```


# 常见报错解决
- HTTPS方式需要密码的解决方案
  [How to fix: fatal: Authentication failed for https://github.com/](https://mycyberuniverse.com/how-fix-fatal-authentication-failed-for-https-github-com.html)


![img](https://img-blog.csdn.net/20171022091942214?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzI1MjA0Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
