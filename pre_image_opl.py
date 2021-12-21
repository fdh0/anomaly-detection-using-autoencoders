# 解压缩
!unzip /content/gdrive/MyDrive/AI_Images/Anormaly/Class8.zip

# 文件夹文件重命名
import os
path = 'KolektorSDD-boxes/Test'
# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
n = 0
for i in fileList:
  # 设置旧文件名（就是路径+文件名）
  oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符
  # 设置新文件名
  newname = path + os.sep + str(n+1) + '.jpg'
  os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
  print(oldname, '======>', newname)
  n += 1


# 压缩文件
!zip -r KolektorSDD-boxes.zip KolektorSDD-boxes

# 储存到Drive
!cp KolektorSDD-boxes.zip /content/gdrive/MyDrive/AI_Images/Anormaly/KolektorSDD.zip

# Resize文件夹图像

!mkdir output
!python resize_folder.py --image KolektorSDD-boxes/Train

!mv KolektorSDD-boxes/Test/*.jpg KolektorSDD-boxes/Test/Label/

# 去除之前的图片
!rm -rf KolektorSDD-boxes/Train/*.jpg
!mv output/* KolektorSDD-boxes/Train/

# 去除之前的图片
!rm -rf KolektorSDD-boxes/Test/*.jpg
!mv output/* KolektorSDD-boxes/Test/


!rm -rf output/*
!python resize_folder.py --image KolektorSDD-boxes/Test

