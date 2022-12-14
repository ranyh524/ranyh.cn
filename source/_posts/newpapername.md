---
title: 字节和位的关系，二者怎么换算
tags:
- 操作系统
- 常识
---

## 字节(Byte/B)
>字节(Byte)是存储数据的基本单位，并且是硬件所能访问的最小单位。cpu只能处理内存中的数据，不能处理硬盘数据。硬盘数据只有调入内存条中才能访问。

## 位(Bit)
>内存里存放的都是二进制代码。内存里面有很多“小格子”，每个“格子”只能存放一个0或1。一个小格子就是一位，所以位要么是0，要么是1，不可能有比位更小的单位。

## 字节和位的关系
>字节是存储数据的基本单位，位是内存中存储数据的最小单位，1字节等于8位

## 为什么字节是硬件所能访问的最小单位
>**硬件是通过地址总线访问内存的，而地址是以字节为单位进行分配的，所以地址总线只能精确到字节。那如何控制到他的某一位呢？这个只能通过"位运算符"，即通过软件的方式来控制。**

## 字节的换算
常见的存储单位主要有bit（位）、B（字节）、KB（千字节）、MB（兆字节）、GB（千兆字节）。它们之间主要有如下换算关系：
>1B=8bit
>1KB=1024B
>1MB=1024KB
>1GB=1024MB


**其中 B 是 Byte 的缩写。**

  比如计算机的内存是 4GB，那么它能存放多少个 0 或 1（即能存放多少位）呢？4×1024×1024×1024×8 位。因为一个 1024 就是 210，所以结果就相当于 32 个 230 这么多！这就是 4GB 内存条所能存储的数据。

硬盘也一样，比如计算机的硬盘是 500GB，那么它能存储的数据为 500×1024×1024×1024×8 位，也就是能存放 4000 个 230 这么多的 0 或 1。

**参考  ：**[字节是什么？如何换算](http://c.biancheng.net/view/140.html)
