---
title: 最全动态规划详解
tags:
- 算法
- c++
---

## 前言
***
`由于动态规划没有固定的套路并且个人认为也是算法里面最难的算法，所以这里讲解动态规划常见的几种模型，因为本文是自己学习后的总结，所以后面还会补充一些细节。`
***如有错误请及时指出***
***


**什么是动态规划**（官方解释，但好像没什么用）
> 	 	 **动态规划 Dynamic Programming，DP**：是运筹学的一个分支，是求解决策过程最优化的过程。20世纪50年代初，美国数学家贝尔曼（R.Bellman）等人在研究多阶段决策过程的优化问题时，提出了著名的最优化原理，从而创立了动态规划。动态规划的应用极其广泛，包括工程技术、经济、工业生产、军事以及自动化控制等领域，并在背包问题、生产经营问题、资金管理问题、资源分配问题、最短路径问题和复杂系统可靠性问题等中取得了显著的效果。


**动态规划的核心思想**
`通俗的讲dp的核心就是记住已经解决过子问题的解，通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。dp常常适用于有重叠子问题和最优子结构性质的问题,动态规划方法所消耗的时间往往远小于朴素解法。`

***分治和动态规划的区别***
>**共同点**：两者都要求原问题具有最优子结构性质，都是将原问题分而治之，分解成若干个规模较小的子问题，然后将子问题的解合并，最终得到答案。
>**不同点**：分治法将分解后的子问题看成相互独立的，通常用递归来做。动态规划将分解后的子问题理解为相互间有联系，有重叠部分，需要记忆，通常用迭代来做。

**动态规划性质**
>1. **最优化原理**：如果问题的最优解所包含的子问题的解也是最优的，就称该问题具有最优子结构，即满足最优化原理
>2. **无后效性**：即某阶段状态一旦确定，就不受这个状态以后决策的影响。也就是说，某状态以后的过程不会影响以前的状态，只与当前状态有关。
>3. **有重叠子问题**：即子问题之间是不独立的，一个子问题在下一阶段决策中可能被多次使用到。（该性质并不是动态规划适用的必要条件，但是如果没有这条性质，动态规划算法同其他算法相比就不具备优势）

**动态规划的步骤**
>1. 确定dp数组及其下标的含义；（找出状态表示）
>2. 确定递推公式；（确定状态转移）
>3. dp数组如何初始化；
>4. 确定遍历顺序；
>5. 举例推导dp数组。

**`求状态表示的小技巧：`**
>**一般由题目要求什么，状态表示就表示成什么。**

**` 求状态转移的小技巧：`**
>**可将所有物品（比如：背包问题中题目给出的物品就是‘物品’，子序列问题中每个数就是物品，股票问题中股票就是物品）看成一个集合，这个集合又是由几个集合组成，然后可以考虑某一个状态表示是由哪几个集合转移来的（前面的集合已经求出）。 举个例子：在背包问题中，可以将所有物品看成一个集合，那么这个集合可能由不同的集合组成，也就是不同的物品组合，然后在目前这个背包容量下可以是由前面哪个集合转移过来才能使价值最大（将哪个物品组装入才能使价值在不超过容量的前提下价值最大）。`要是看不懂可以将后面背包问题看了在回来看这儿`**

**哪种题型一般使用动态规划求解**

>1. 最优解问题：数组中最大值型，比如：最长上升子序列，最大子数组，最长公共子序列等问题。
>2. 求可行性问题：如果有这样一个问题，让你判断是否存在一条总和为 x 的路径（如果找到了，就是 True；如果找不到，自然就是 False），或者让你判断能否找到一条符合某种条件的路径，那么这类问题都可以归纳为求可行性问题，并且可以使用动态规划来解。
>3. 求方案数问题：求方案总数也是比较常见的一类动态规划问题。比如说给定一个数据结构和限定条件，让你计算出一个方案的所有可能的路径，那么这种问题就属于求方案总数的问题。
>

## 数字三角形模型
$~~~~$
#### 1. 数字三角形
![在这里插入图片描述](https://img-blog.csdnimg.cn/d3a97f66e09c4f3b881643bed8a7ac6a.png)

**解题思路**
>可将题目中所有元素看成一个集合，题目中要求路径中的数字和最大，并且题目中的元素的二维排列的，那么`状态表示就可以是f[i][j]表示沿着某条路径走到第i行第j列这条路径上的数字和为多少`，那状态转移怎么算呢？就像我们上面说的，**我们可以看第i行第j列这个状态可能由哪些集合转移过来，可以看到第i行第j列可以由第i-1行第j列和第i-1行第j-1列转移而来，而第i-1行第j列和第i-1行第j-1列这两个状态可以表示成f[i-1][j],f[i-1][j-1]（也就是上面提到的集合），虽然f[i][j]可以由f[i-1][j-1]表示，但是前提是这两个状态（也可以说成这两个集合）得求出来**。
>
>`状态表示：f[i][j]表示从左上角走到第i行第j列的和的最大值`
>`转移方程：f[i][j] = max(f[i-1][j-1],f[i-1][j])`

**此模型也可转换为最长路径问题**

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N = 510,INF = 0x3f3f3f3f;
int a[N][N],f[N][N];

int main()
{
	int n;
	cin>>n;
	for(int i=1;i<=n;i++)
		for(int j=1;j<=i;j++)
			cin>>a[i][j];
	//先将f数组初始化
	for(int i=0;i<=n;i++)
		for(int j=0;j<=i+1;j++)
			f[i][j] = -INF;
	//第一个点走到第一个点的最大值就是它本身
	f[1][1] = a[1][1];
	for(int i=2;i<=n;i++)
		for(int j=1;j<=i;j++)
			f[i][j] = max(f[i-1][j-1],f[i-1][j])+a[i][j];
	int res = 0;
	for(int i=1;i<=n;i++) res = max(res,f[n][i]);
	cout<<res<<endl;
	return 0;
}
```

#### 2. 最低通行费
![在这里插入图片描述](https://img-blog.csdnimg.cn/8eee687be98842779257cf3a0510a2d8.png)

***解题思路***
>同样可以把所有小方格看成一个集合，题目中求得是规定时间内穿过的最小费用，那么`状态表示`我们就可以表示成f[i][j]为走到第i行第j列所需要的费用，那这个状态怎么来的呢，**由题意可看出可以由左上和正上方走下来，所以这两个可以用状态表示这两个集合f[i-1][j],f[i-1][j-1]**。

`由题目总的时间<=2N-1可知，商人必须从上往下走或走到右下角`
`状态表示：f[i][j]表示走到第i行第j列所需要的费用`
`状态转移：f[i][j] = max(f[i-1][j],f[i-1][j-1])+a[i][j]`

**代码如下**

```c
#include<bits/stdc++.h>
using namespace std;
const int N=110;
int a[N][N],dp[N][N];
int n;

int main()
{
    cin>>n;
    for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
            cin>>a[i][j];
            
    for(int i=1;i<=n;i++) dp[i][1]=dp[i-1][1]+a[i][1];
    for(int i=2;i<=n;i++) dp[1][i]=dp[1][i-1]+a[1][i];
    
    for(int i=2;i<=n;i++)
        for(int j=2;j<=n;j++)
            dp[i][j]=min(dp[i-1][j],dp[i][j-1])+a[i][j];
            
    cout<<dp[n][n]<<endl;
    return 0;
}

```

#### 3. 方格取数

![在这里插入图片描述](https://img-blog.csdnimg.cn/833cb4a923fb4afbbb32edb2c68deb71.png)
***解题思路***
>`注意题目中说：此人从 AA 点到 BB 点共走了两次，所以我们可以设两条路径是同时出发的所以两条路径时很容易想到维护当前两条路径的状态（坐标）也就是 f[x1][y1][x2][y2]当然，这样做空间复杂度会达到 n^4，所以我们需要优化由我们同时出发的条件，我们注意到：k=x1+y1=x2+y2,k=x1+y1=x2+y2
>所以我们可以只用三个维度 f[k][x1][x2] 来维护就行了！（重点），因为 y1=k−x1,y1=k−x1，y2=k−x2,y2=k−x2
>最后，按照数字三角形模型将两条路径的转移合并在一起（一共四种）即可解决`
>**以上坐标的处理方式的技巧对减小时间复杂度很常用**

`状态表示：f[k][i][j] 表示路径长度为k,第一条路线到x1=i,第二条路线到x2=j的所有方案`
`f(k,i,j)=max{f(k−1,i,j),f(k−1,i−1,j),f(k−1,i,j−1),f(k−1,i−1,j−1)}+w`

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N=20;
int g[N][N],dp[2*N][N][N];
int n;

int main()
{
    cin>>n;
    int a,b,c;
    while(cin>>a>>b>>c,a||b||c) g[a][b]=c;
    for(int k=1;k<=2*n;k++)
    {
        for(int i1=1;i1<=n;i1++)
        {
            for(int i2=1;i2<=n;i2++)
            {
                int j1=k-i1,j2=k-i2;
                if(j1>=1&&j1<=n&&j2>=1&&j2<=n)
                {
                    int t=g[i1][j1];
                    if(i2!=i1) t+=g[i2][j2]; 
                    int &x = dp[k][i1][i2];
                    x = max(x,dp[k-1][i1-1][i2-1]+t);
                    x = max(x,dp[k-1][i1-1][i2]+t);
                    x = max(x,dp[k-1][i1][i2-1]+t);
                    x = max(x,dp[k-1][i1][i2]+t);
                }
            }
        }
    }
    cout<<dp[2*n][n][n]<<endl;
    return 0;
}
```
#### 总结
>**此类问题比较简单，问法通常是求在一个矩阵中从起点到终点最小/最大的花费，状态转移方程也是固定的模型：f[i][j] = max(f[i-1][j],f[i-1][j-1]) (看具体情况)**


## 最长上升子序列模型
$~~~~$
#### 1. 最长上升子序列（LIS）
![在这里插入图片描述](https://img-blog.csdnimg.cn/34dd5b53adac4794ba3b50b1ea7d141d.png)

**解题思路**

**注意子序列和子串的区别：子序列就是子串随意删除几个字符（不连续），子串就是连续的一串字符**。所以本题是要遍历子串的，如果题目是数组的话可以使用双指针做。

`状态表示：f[i]表示以A[i]结尾的LIS的长度`
`转移方程：f[i] =max(f[j]+1,f[i])  (A[i]之前子序列的最大长度+1,不一定是以前一个结尾的子序列的最大值+1)`

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N = 1010;
int a[N],f[N],res;
int main()
{
	int n;
	cin>>n;
	for(int i=1;i<=n;i++)
		cin>>a[i];
	memset(f,1,sizeof f);
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<i;j++)
		{
			if(a[i]>a[j]) f[i] = max(f[j]+1,f[i]);
		}
		res = max(res,f[i]);
	}
	cout<<res<<endl;
	return 0;
}

```
#### 2. 怪盗基德的滑翔翼
![在这里插入图片描述](https://img-blog.csdnimg.cn/7a5eebb77f384ddd84cf02e412365d98.png)
***解题思路***

`同上题，做两遍LIS`
`状态表示：f[i]表示以A[i]结尾的LIS长度`
`状态转移：f[i] = max(f[i-1]+1,f[i]) 同上`

**代码如下**

```c
#include<bits/stdc++.h>
using namespace std;
const int N=110;
int g[N],dp[N];
int k;

int main()
{
    cin>>k;
    while(k--)
    {
        int n,res=0;
        cin>>n;
        for(int i=1;i<=n;i++) cin>>g[i];
        
        for(int i=1;i<=n;i++)
        {
            dp[i] = 1;
            for(int j=1;j<i;j++)
                if(g[i]>g[j])
                    dp[i] = max(dp[i],dp[j]+1);
            res = max(res,dp[i]);
        }
            
        memset(dp,0,sizeof dp);    
        
        for(int i=n;i>=1;i--)
        {
            dp[i]=1;
            for(int j=n;j>i;j--)
                if(g[i]>g[j])
                    dp[i] = max(dp[i],dp[j]+1);
            res = max(res,dp[i]);
        }
        
        cout<<res<<endl;
    }
    return 0;
}
```



#### 3. 最长公共子序列
![在这里插入图片描述](https://img-blog.csdnimg.cn/c3ecdbcef83448508e92743c608e1fdb.png)

***解题思路***

`状态表示：f[i][j]表示以a[i],b[j]结尾的字符串的最长公共子序列`
`状态转移：f[i][j] = max(f[i][j-1),f[i-1][j])`


**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N=1010;
char a[N],b[N];
int f[N][N];
int n,m;

int main()
{
    cin>>n>>m;
    cin>>a+1>>b+1;
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        {
        	//如果a[i]!=b[j]
            f[i][j]=max(f[i][j-1],f[i-1][j]);
            //如果a[i]==b[j],那就是i-1,j-1结尾的最长公共子序列+1
            if(a[i]==b[j]) f[i][j]=max(f[i][j],f[i-1][j-1]+1);
        }
    cout<<f[n][m]<<endl;
    return 0;
}

```

#### 4. 最长公共上升子序列
![在这里插入图片描述](https://img-blog.csdnimg.cn/34c1eb25f5164817950abf9cd1f0da01.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/7778b6fcc93c4cbe99c951887aef6099.png)
***题目分析***
`状态表示：f[i][j]代表所有a[1 ~ i]和b[1 ~ j]中以b[j]结尾的公共上升子序列的集合`
`状态转移：`
>**首先依据公共子序列中是否包含a[i]，将f[i][j]所代表的集合划分成两个不重不漏的子集：
>不包含a[i]的子集，最大值是f[i - 1][j]；
>包含a[i]的子集，将这个子集继续划分，依据是子序列的倒数第二个元素在b[]中是哪个数：
>子序列只包含b[j]一个数，长度是1；
>子序列的倒数第二个数是b[1]的集合，最大长度是f[i - 1][1] + 1；
>…
>子序列的倒数第二个数是b[j - 1]的集合，最大长度是f[i - 1][j - 1] + 1；
>如果直接按上述思路实现，需要三重循环;
>然后我们发现每次循环求得的maxv是满足a[i] > b[k]的f[i - 1][k] + 1的前缀最大值。
>因此可以直接将maxv提到第一层循环外面，减少重复计算，此时只剩下两重循环。
>最终答案枚举子序列结尾取最大值即可。**

**代码如下**
```c
#include <bits/stdc++.h>

using namespace std;

const int N = 3010;

int n;
int a[N], b[N];
int f[N][N];
int g[N][N];  

int main() {

    cin >> n;
    for (int i = 1; i <= n; ++ i) cin >> a[i];
    for (int j = 1; j <= n; ++ j) cin >> b[j];


    for (int i = 1; i <= n; ++ i) {
        int maxv = 1;
        for (int j = 1; j <= n; ++ j) {
            f[i][j] = f[i - 1][j];
            if (a[i] == b[j]) f[i][j] = max(f[i][j], maxv);
            if (a[i] > b[j]) maxv = max(maxv, f[i - 1][j] + 1);
        }
    }

    int ans = 0;
    for (int i = 1; i <= n; ++ i) ans = max(f[n][i], ans);

    cout << ans << endl;
    return 0;
}
```
#### 总结
> **LIS类型dp的状态表示也是比较好想出来的，同样其他这种类型的题目的状态转移方程和状态表示和上面所有LIS类型题很类似**


## 背包问题模型
$~~~~$
`在背包模型开始前先可以参考一下‘背包九讲’这个博客，写的非常详细，非常好，非常推荐`
[背包九讲:https://blog.csdn.net/yandaoqiusheng/article/details/84782655?](https://blog.csdn.net/yandaoqiusheng/article/details/84782655?)
`以下所列背包问题都是常见背包问题以及所涉及的证明及时间复杂度的优化就不赘述，直接可以看上面博客，因为很详细。`

**背包问题状态转移的核心就是`第i个物品选还是不选`**
$~~~~$
#### 01背包
##### 1. 01背包
![在这里插入图片描述](https://img-blog.csdnimg.cn/ba8a4c68b68f486ab73d8198d530087c.png)
***解题思路***

`状态表示：f[i]表示在背包容量不超过j时的最大价值，此一维状态是由二维压缩而来，具体过程可转上述‘背包九讲’链接`
`状态转移：f[i] = max(f[i],f[i-v[i]]+w[i]`

>***01背包模型中一种物品只能使用一次，一般只要是求在不超过某个限制条件下价值的最大值都可尝试使用01背包问题***

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N=1005;
int f[N],v[N],w[N];//w[i]是价值，v[i]是体积

int main()
{
	int n,m;
	cin>>n>>m;//n件物品和体积限制
	for(int i=1;i<=n;i++)
		cin>>v[i]>>w[i];
	for(int i=1;i<=n;i++)
		for(int j=m;j>=v[i];j--)
			f[j] = max(f[j],f[j-v[i]]+w[i]);
	cout<<f[m]<<endl;
	return 0;
}
```
##### 2. 装箱问题
![!\](https://img-blog.csdnimg.cn/631e456d0b5143c793f9adbdc5624334.png)](https://img-blog.csdnimg.cn/178d4fab2b70400bba63e2953ce4f3ae.png)

***题目分析***
`本题可以把箱子的体积看成背包问题的体积和价值，所以状态表示及状态转移同上。`

**代码如下**
```c
#include<bits/stdc++.h>
using  namespace std;
const int N = 21000;
int f[N],w[N];
int v,n;//最大体积和箱子个数

int main()
{
	cin>>v>>n;
	for(int i=1;i<=n;i++) cin>>w[i];
	for(int i=1;i<=n;i++)
		for(int j=v;j>=w[i];j--)
			f[j] = max(f[j],f[j-w[i]]+w[i]);
	cout<<v-f[v]<<endl;
	return 0;
}

```
####  完全背包
##### 1. 完全背包问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/e3588fee2e694939b572e145928eb9e1.png)
***问题分析***
>**完全背包问题中，同样也是求将若干种物品放入一个容量一定的容器，求最大的‘价值’，但和01背包问题的区别是前者一种物品只能使用一次，后者每种物品由无数件可以使用**

`状态表示：f[n]表示容量为i时的最大价值`
`状态转移：f[i] =  max(f[i-v[i]]+w[i]) 但是要和01背包问题区分开，两者虽然在状态转移方程一样，但是在枚举体积时的顺序不一样`

***
***给出简单的由二维优化到此一维的推导***

>**我们列举一下更新次序的内部关系:**
>`f[i , j ] = max( f[i-1,j] , f[i-1,j-v]+w ,  f[i-1,j-2*v]+2*w , f[i-1,j-3*v]+3*w , .....)`
>`f[i , j-v]= max(            f[i-1,j-v]   ,  f[i-1,j-2*v] + w , f[i-1,j-3*v]+2*w , .....)`
>**由上两式，可得出如下递推关系：**
>                  `f[i][j]=max(f[i,j-v]+w , f[i-1][j]) `
>**有了上面的关系，那么其实k循环可以不要了，核心代码优化成这样:**
>```c
>for(int i = 1 ; i <=n ;i++)
>for(int j = 0 ; j <=m ;j++)
>{
>f[i][j] = f[i-1][j];
>if(j-v[i]>=0)
>    >  f[i][j]=max(f[i][j],f[i][j-v[i]]+w[i]);
>}
>```
>**这个代码和01背包的非优化写法很像啊!!!我们对比一下，下面是01背包的核心代码**
>```c
>for(int i = 1 ; i <= n ; i++)
>for(int j = 0 ; j <= m ; j ++)
>{
> f[i][j] = f[i-1][j];
>if(j-v[i]>=0)
>    >  f[i][j] = max(f[i][j],f[i-1][j-v[i]]+w[i]);
>}
>```
>**两个代码其实只有一句不同**
>`f[i][j] = max(f[i][j],f[i-1][j-v[i]]+w[i]);//01背包`
>`f[i][j] = max(f[i][j],f[i][j-v[i]]+w[i]);//完全背包问题`
>**因为和01背包代码很相像，我们很容易想到进一步优化。核心代码可以改成下面这样**
>```c
>for(int i = 1 ; i<=n ;i++)
>for(int j = v[i] ; j<=m ;j++)//注意了，这里的j是从小到大枚>举，和01背包不一样
>   {
>         f[j] = max(f[j],f[j-v[i]]+w[i]);
>}
>```
>***综上所述，完全背包的最终写法如下：***

```c
#include<iostream>
using namespace std;
const int N = 1010;
int f[N];
int v[N],w[N];
int main()
{
    int n,m;
    cin>>n>>m;
    for(int i = 1 ; i <= n ;i ++)
    {
        cin>>v[i]>>w[i];
    }

    for(int i = 1 ; i<=n ;i++)
	    for(int j = v[i] ; j<=m ;j++)
	    {
	            f[j] = max(f[j],f[j-v[i]]+w[i]);
	    }
    cout<<f[m]<<endl;
    return 0;
}
```
##### 2. 买书
![在这里插入图片描述](https://img-blog.csdnimg.cn/f0cdb5beddbd4e88995579abdf96538e.png)
***题目分析***

`本题很容易就可以看出来时完全背包问题，所以状态转移和状态表示一样，具体情况看注释`
***本题同样也是背包问题求方案数的一种题型***

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N = 1100;
int a[4] = {10,20,50,100};
//f[N]表示钱数为n时的方案数
//f[j] = f[j-a[i]]+f[j];第i个物品选不选
int f[N],n;

int main()
{
	cin>>n;
	f[0] = 1;
	for(int i=0;i<n;i++)
	{
		for(int j=a[i];j<=n;j++)
			f[j] = f[j]+f[j-a[i];
	}
	cout<<f[n]<<endl;
	return 0;
}
```

####  多重背包Ⅰ
![在这里插入图片描述](https://img-blog.csdnimg.cn/7e85d6a7537e49e28ca4f402b8a53551.png)
***题目分析***
>**多重背包问题也是求将若干种物品放入容量固定的容器求最值，和前两种区别是此问题是每一种物品的个数是不确定的，因为每一种物品的个数未知，所以没有办法像前两种算法那样优化为一维。**

`状态表示：f[i]][j]表示从前i个物品中选,且总体积不超过j的所有方案的集合`
`状态转移：f[i][j] = max(f[i][j], f[i - 1][j - k * v[i]] + k * w[i]);//和完全背包问题的朴素代码一样`

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N = 110;
int v[N],w[N],s[N];
int f[N][N];

int main()
{
	int n,m;
	cin>>n>>m;
	for(int i=1;i<=n;i++) cin>>v[i]>>w[i]>>s[i];
	//先枚举种数
	for(int i=1;i<=n;i++)
		//然后枚举体积，注意，这里不能从v[i]开始枚举
		for(int j=0;j<=m;j++)
		{
			//最后枚举第i种物品的个数
			for(int k=0;k*v[i]<=j&&k<=s[i];k++)
				f[i][j] = max(f[i][j],f[i-1][j-k*v[i]]+k*w[i]);
		}
	cout<<f[n][m]<<endl;
	return 0;	
}
```

####  多重背包Ⅱ

![在这里插入图片描述](https://img-blog.csdnimg.cn/d0b7f0a9a881442c8e19829079e4a1d0.png)
***题目分析***

`因为上面的多重背包的解法的时间复杂度是n^3,当数据范围达到1000时就会超时，所以背包问题Ⅱ是针对数据范围达到1000时的做法采用了二进制优化。`
$~~~~$**`选看`**

**代码如下**
```c

#include<bits/stdc++.h>
using namespace std;
const int N=12010;
int w[N],v[N];
int f[N];

int main()
{
    int n,m;
    int cnt=0;
    cin>>n>>m;
    for(int i=1;i<=n;i++)
    {
        int a,b,s;
        cin>>a>>b>>s;
        int k=1;
        while(k<=s)
        {
            cnt++;
            v[cnt]=a*k;
            w[cnt]=b*k;
            s-=k;
            k*=2;
        }
        if(s>0)
        {
            cnt++;
            w[cnt]=b*s;
            v[cnt]=a*s;
        }
    }
    n=cnt;
    for(int i=1;i<=n;i++)
        for(int j=m;j>=v[i];j--)
            f[j]=max(f[j],f[j-v[i]]+w[i]);
    cout<<f[m]<<endl;
    return 0;
}
```

####  分组背包
![在这里插入图片描述](https://img-blog.csdnimg.cn/ef472e013d804f0bb7a8a4290df5cfd5.png)
***题目分析***
>**是求将若干种物品放入容量固定的容器求最值，此问题是每一组物品的物品只能选一个，每一组物品的个数是不确定的，每一组其中每一件的价值和体积也是不一样的。但是可以将每一个组看成一个整体，因为每一组只能选一个物品，就可以使用01背包解决**

`因为可以当01背包来做，所以状态转移和状态表示一样和01背包`

**代码如下**

```c
#include<bits/stdc++.h>
using namespace std;
const int N=110;
int v[N][N],w[N][N];//第i组第j个物品的体积和价值
int s[N];//第i组物品的数量
int f[N];

int main()
{
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++)
    {
        cin>>s[i];
        for(int j=0;j<s[i];j++)
            cin>>v[i][j]>>w[i][j];
    }
    
    //枚举物品组
    for(int i=1;i<=n;i++)
    	//枚举体积
        for(int j=m;j>=0;j--)
        	//枚举决策，也就是选这个物品组的哪个物品
            for(int k=0;k<s[i];k++)
                if(v[i][k]<=j)
                    f[j]=max(f[j],f[j-v[i][k]]+w[i][k]);
    cout<<f[m]<<endl;
    return 0;
}

```

#### 有依赖的背包问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/5f9051e3629743dca08575573e364403.png)
***题目分析***
>**这种背包问题的描述很类似与拓扑排序，都是相互之间有依赖关系的，选一个物品之前必须先选择另外一个和他有关系的物品。**

***题解分析***
>**根据题设的拓扑结构可以观察出每个物品的关系构成了一棵树,而以往的背包DP每个物品关系是任意的（但我们一般视为 `线性`的）所以，这题沿用背包DP的话，要从原来的线DP改成 树形DP 即可。然后思考 `树形DP`（后面会介绍）的状态转移。
>先比较一下以往线性背包DP的状态转移，第 i 件物品只会依赖第 i−1件物品的状态。
>如果本题我们也采用该种状态依赖关系 的话，对于节点 i，我们需要枚举他所有子节点的组合` 2^k` 种可能
>再枚举 体积，最坏时间复杂度 可能会达到 `O(N×2^N×V)`（所有子节点都依赖根节点)毫无疑问会 TLE。因此我们需要`枚举每个状态分给各个子节点的体积。这样时间复杂度就是 O(N×V×V)`
>具体分析如下：**
>`状态表示：f[i][j]表示考虑第i个物品为根节点的子树，且选上i选法的总体积不超过j的方案。`
>`状态转移：f[u[[j] = max(f[u][j],f[u][j-k]+f[son][k]);`

**代码如下**

```c
#include<bits/stdc++.h>

using namespace std;

const int N = 110;

int n, m, root;
int h[N], e[N], ne[N], idx;
int v[N], w[N];
int f[N][N];

//图的存储
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}
void dfs(int u)
{
	//其实是一个分组背包问题
    //先枚举所有状态体积小于等于j-v[u]的所有子节点们能够获得的最大价值
    for (int i = h[u]; ~i; i = ne[i])
    {
    	//先取出子节点
        int son = e[i];
        dfs(son); //从下往上算，先计算子节点的状态
        for (int j = m - v[u]; j >= 0; -- j) //枚举所有要被更新的状态
        {
            for (int k = 0; k <= j; ++ k)   //枚举该子节点在体积j下能使用的所有可能体积数
            {
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k]);
            }
        }
    }
    //选上第u件物品
    for (int j = m; j >= v[u]; -- j) f[u][j] = f[u][j - v[u]] + w[u];
    for (int j = 0; j <  v[u]; ++ j) f[u][j] = 0;   //清空没选上u的所有状态
}
int main()
{
    memset(h, -1, sizeof h);
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i)
    {
        int p;
        cin >> v[i] >> w[i] >> p;
        if (p == -1) root = i;
        else add(p, i);
    }
    //不管选择哪个物品都会在选择根结点，所以找出根节点从根节点开始dfs
    dfs(root);
    cout << f[root][m] << endl;
    return 0;
}

```

#### 背包问题求方案数
![在这里插入图片描述](https://img-blog.csdnimg.cn/a4d4a8a274794ac096e416ac80f00390.png)
***题目分析***

>题目要求输出字典序最小的解，假设存在一个包含第1个物品的最优解，为了确保字典序最小那么我们必然要选第一个。那么问题就转化成从2～N这些物品中找到最优解。之前的f(i,j)记录的都是前i个物品总容量为j的最优解，那么我们现在将f(i,j)定义为从第i个元素到最后一个元素总容量为j的最优解。接下来考虑状态转移：
>`f(i,j)=max(f(i+1,j),f(i+1,j−v[i])+w[i])`
>两种情况，第一种是不选第i个物品，那么最优解等同于从第`i+1`个物品到最后一个元素总容量为j的最优解；第二种是选了第i个物品，那么最优解等于当前物品的价值w[i]加上从第`i+1`个物品到最后一个元素总容量为`j−v[i]`的最优解。
>计算完状态表示后，考虑如何的到最小字典序的解。首先`f(1,m)`肯定是最大价值，那么我们便开始考虑能否选取第1个物品呢。
>如果`f(1,m)=f(2,m−v[1])+w[1]`，说明选取了第1个物品可以得到最优解。
>如果`f(1,m)=f(2,m)`，说明不选取第一个物品才能得到最优解。
>如果`f(1,m)=f(2,m)=f(2,m−v[1])+w[1]`，说明选不选都可以得到最优解，但是为了考虑字典序最小，我们也需要选取该物品。

`因为要求具体的方案数，所以就不能使用优化后的一维dp，而是要用二维，`

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N=1010;
int f[N][N];
int n,m;
int v[N],w[N];

int main()
{
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>v[i]>>w[i];
    for(int i=n;i>=1;i--)
        for(int j=0;j<=m;j++)
        {
            f[i][j]=f[i+1][j];
            if(j>=v[i]) f[i][j] = max(f[i][j],f[i+1][j-v[i]]+w[i]);
        }
        
    int j=m;
    for(int i=1;i<=n;i++)
        if(j>=v[i]&&f[i][j]==f[i+1][j-v[i]]+w[i])
        {
            cout<<i<<" ";
            j-=v[i];
        }
    return 0;
}
```
#### 总结
`记忆小技巧`
>**几种背包问题第一维都是枚举种数，第二维都是枚举体积。
>要是一种物品只有一件就从大到小枚举体积，否则就从小到大枚举体积。**

`01背包模型模板`
>***一般是求将若干种物品在满足一定条件下放入一个大小一定的‘箱子’，且一种物品只能使用一次***
>```c
>//先枚举物品个数
>for(int i=0;i<n;i++)
>	//再枚举物品体积，从大到小
>	for(int j=v;j>=0;j--)
>		f[j] = min(f[j],f[j-w[i]]+v[i]);
>cout<<f[v];//f[v]就是最后答案
>```

`完全背包问题模板`
>***一般是求将若干种物品在满足一定条件下放入一个大小一定的‘箱子’，且一种物品可以放多次，和01背包问题有点像，但是注意枚举体积时的顺序***
>```c
>for(int i=1;i<=n;i++)
>	for(int j=w[i];j<=v;j++)
>		f[j] = max(f[j],f[j-w[i]]+v[i]);
>cout<<f[v];//最后的答案

`多重背包问题模板`
>**多重背包问题也是求将若干种物品放入容量固定的容器求最值，和前两种区别是此问题是每一种物品的个数是不确定的，但并不是无限多个。因为每一种物品的个数未知，所以没有办法像前两种算法那样优化为一维。**
>```c
>for(int i=0;i<n;i++)
>	for(int j=0;j<=m;j++)
>		for(int k=0;k<=s[i]&&k*v[i]<=j;k++)
>			f[i][j] = max(f[i][j],f[i-1][j-k*v[i]]+k*w[i]);

`分组背包问题模板`
>**是求将若干种物品放入容量固定的容器求最值，此问题是每一组物品的物品只能选一个，每一组物品的个数是不确定的，每一组其中每一件的价值和体积也是不一样的。但是可以将每一个组看成一个整体，因为每一组只能选一个物品，就可以使用01背包解决**
>```c
>//枚举物品组
>for(int i=0;i<n;i++)
>	//枚举体积,从大到小
>	for(int j=m;j>=v[i];j--)
>		//枚举决策
>		for(int k=0;k<s[i];k++)
>			f[j] = max(f[j],f[j-v[i][k]]+w[i][j]);
>cout<<f[m]<<endl;
## 区间dp模型
$~~~~$
#### 1. 石子合并
![在这里插入图片描述](https://img-blog.csdnimg.cn/36178f6e31684830913156fa0bbef79a.png)
***题目分析***

`状态表示：f[i][j]表示区间i~j之间的最值`
`状态转移：min(dp[i][j], dp[i][k] + dp[k + 1][j] + w[i][j])      //k是ij区间的中间分割点;`

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N = 310;
int s[N];
int f[N][N];
int main()
{
	int n;
	cin>>n;
	for(int i=1;i<=n;i++)cin>>s[i];
	for(int i=1;i<=n;i++) s[i] += s[i-1];
	for(int len = 2;len<=n;len++)
	{
		for(int i=1;i+len-1<=n;i++)
		{
			int l=i,r = len+i-1;
			f[l][r] = 1e9;
			for(int k=l,k<r;k++)
				f[l][r] = max(f[l][r],f[l][k]+f[k+1][r]+s[r]-s[l-1];
		}
	}
	cout<<f[1][n]<<endl;
	return 0
}
```
#### 2. 加分二叉树
![在这里插入图片描述](https://img-blog.csdnimg.cn/68479579ae5d41ec9e7a0fa897e3fac8.png)
***题目分析***

>**因为二叉树向下映射（将二叉树向下做投影）得到的序列就是二叉树的中序遍历，所以正是因为二叉树的中序遍历的这个特点就使得我们想到了可以使用区间dp。**

`状态表示：f[i][j]表示中序遍历由i号结点到j号结点这个范围的最值`
`状态转移：f[i][j] = f[i][k-1]*f[k+1][j]+w[k];也就是在根节点为k的时候取得最值`

**代码如下**
```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 40;
//g用来记录l,r区间的根节点
int f[N][N],g[N][N],a[N][N];
int n;
//前序遍历
void dfs(int l,int r)
{
    if(l>r) return ;
    cout<<g[l][r]<<" ";
    dfs(l,g[l][r]-1);
    dfs(g[l][r]+1,r);
}
int main()
{
	cin>>n;
	for(int i=1;i<=n;i++) cin>>a[i];
	for(int len=1;len<=n;len++)
    {
        for(int l=1;len+l-1<=n;l++)
        {
            int r = len+l-1;
            if(l == r) 
            {
                f[l][r] = a[l];
                g[l][r] = l;
            }
            else
            {
                for(int k=l;k<=r;k++)
                {
                    int left = k==l?1:f[l][k-1];//求左子树
                    int right = k==r?1:f[k+1][r];//求右子树
                    int s = left*right+a[k];//计算分数
                    if(f[l][r]<s)
                    {
                        f[l][r] = s;
                        g[l][r] = k;
                    }
                }
            }
        }
    }
    cout<<f[1][n]<<endl;
    dfs(1,n);
    return 0;
}


```

#### 总结
`区间dp问题模板`
>所有的区间dp问题枚举时，第一维通常是枚举区间长度，并且一般 len = 1 时用来初始化，枚举从 len = 2 开始；第二维枚举起点 i （右端点 j 自动获得，j = i + len - 1）
>```c
>for (int len = 1; len <= n; len++) {         // 区间长度
>for (int i = 1; i + len - 1 <= n; i++) { // 枚举起点
>int j = i + len - 1;                 // 区间终点
>   >     if (len == 1) {
>     >     dp[i][j] = 初始值
>       >     continue;
>      >}
>      >for (int k = i; k < j; k++) {        // 枚举分割点，构造状态转移方程
>    >      dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + w[i][j]);
>   }
> }
>}


## 状态机模型
$~~~~$
`状态机特点：描述的是一个过程，不是一个结果。比如：在买股票的过程中买入股票和卖出股票就是一个过程只有在卖出后才可以计算收益。而不像背包问题就只有放和不放着一种情况，只是一种结果。`

**`股票问题就是一种很经典的状态机模型`**


#### 1. 大盗阿福
![在这里插入图片描述](https://img-blog.csdnimg.cn/8be074f86b0d47d18f30e3ccb51c1f05.png)
***题目分析***
>**本题如果用背包问题考虑的话很容易可以得出`状态表示：f[i]表示偷窃前i家店铺所得到的最大财宝。状态转移：f[i] = max(f[i-1],f[i-2]+a[i])//分别对应选前一个和不选前一个`。但是会发现这个状态转移涉及到i-2层，并且如果有个题状态涉及到前三层，前n层的话就很难做，所以可以考虑使用状态机：直接将这个点的所有状态表示出来。**

`状态表示：f[i][1]表示第i个店铺抢，f[i][0]表示第i个店铺不抢`
`状态转移：f[i][0] = max(f[i-1][1],f[i-1][0])，f[i][1] = f[i-1][0]+a[i]`

**代码如下**
```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 100010,INF = 0x3f3f3f3f;
int f[N][2],a[N];
int t,n;

int main()
{
    cin>>t;
    while(t--)
    {
        cin>>n;
        for(int i=1;i<=n;i++) cin>>a[i];
        f[0][0] = 0;f[0][1] = -INF;
        for(int i=1;i<=n;i++)
        {
            f[i][0] = max(f[i-1][0],f[i-1][1]);
            f[i][1] = f[i-1][0]+a[i];
        }
        cout<<max(f[n][1],f[n][0])<<endl;
    }
    return 0;
}

```

#### 2. 股票买卖Ⅳ
![在这里插入图片描述](https://img-blog.csdnimg.cn/7417d68d1d7f4bde90d3c0c5e11bf9da.png)

***题目分析***
>**分析方法和上题一样**

`状态表示：f[i][j][k]表示目前是第i天，且已完成j笔完整交易，并且当前持股(k=1)或未持股(k=0)`
`状态转移：如果第i天持有股，那么既可以由第i−1天持有股没有交易，也可以由第i-1天空仓然后第i天买入，即:f(i,j,1)=max{f(i−1,j,1),f(i−1,j,0)−a[i])};如果第i天未持股，那么即可以由第i−1天未持股，也可以卖出，即f(i,j,0)=max{f(i−1,j,0),f(i−1,j−1,1)+a[i]);`

![在这里插入图片描述](https://img-blog.csdnimg.cn/534bb5c3a62a470098b0385eae6909f2.png)
**代码如下**
```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 100010,INF = 0x3f3f3f3f,M=110;
int f[N][M][2],a[N];
int n,k;

int main()
{
    cin>>n>>k;
    for(int i=1;i<=n;i++)   cin>>a[i];
    
    memset(f,-0x3f,sizeof f);
    for(int i=0;i<=n;i++) f[i][0][0] = 0;
    
    for(int i=1;i<=n;i++)
        for(int j=1;j<=k;j++)
        {
            f[i][j][0] = max(f[i-1][j][0],f[i-1][j][1]+a[i]);
            f[i][j][1] = max(f[i-1][j][1],f[i-1][j-1][0]-a[i]);
        }
        
    int res=0;
    for(int i=0;i<=k;i++) res = max(f[n][i][0],res);
    cout<<res<<endl;
    return 0;
        
}
```
`优化为一维的做法，就相当于第i天的状态只有持仓和空仓两种状态`

**代码如下**

```c++
#include <bits/stdc++.h>

using namespace std;
const int N = 100010, M = 110, INF = 0x3f3f3f3f;
int n, m;
int w[N];
int f[M][2];
int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);

    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
        {
            if(i == 1) f[j][1] = -w[1];
            else
            {
                f[j][0] = max(f[j][0], f[j][1] + w[i]);
                f[j][1] = max(f[j][1], f[j - 1][0] - w[i]);

            }            
        }
    cout << f[m][0];

    return 0;
}
```
#### 3.股票买卖Ⅴ
![在这里插入图片描述](https://img-blog.csdnimg.cn/cc87d9e4a47e433bab4d7d3dd52fe6e8.png)

***题目分析***
>**本题可以将第i天分为空仓期(k=0), 冷冻期(k=2),持仓期(k=1)**

1. 如果第 i 天是 **空仓 (k=0)** 状态，则**i-1**天可能是**空仓** (k=0) 或 **冷冻期 (k=2)** 的状态
2. 如果第 i 天是**冷冻期 (k=2)** 状态，则**i-1**天只可能是**持仓 (k=1)** 状态，在第 i 天选择了 卖出
3. 如果第 i 天是**持仓 (k=1)** 状态，则 i-1 天可能是**持仓 (k=1)** 状态 或 **空仓(k=0)**的状态 （买入）`

`状态转移:f[i][0] = max(f[i-1][0],f[i-1][2]-a[i]);
        f[i][1] = max(f[i-1][0]+a[i],f[i-1][2]);
        f[i][2] = max(f[i-1][1],f[i-1][2]);`

**代码如下**
```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 100010,INF = 0x3f3f3f3f;
int f[N][3],a[N];
int n;

int main()
{
    cin>>n;
    for(int i=1;i<=n;i++) cin>>a[i];
    
    f[0][0] = f[0][1] = -INF;
    f[0][2] = 0;
    
    for(int i=1;i<=n;i++)
    {
        f[i][0] = max(f[i-1][0],f[i-1][2]-a[i]);
        f[i][1] = max(f[i-1][0]+a[i],f[i-1][2]);
        f[i][2] = max(f[i-1][1],f[i-1][2]);
    }
    
    cout<<max(f[n][1],f[n][2])<<endl;
    return 0;
}

```

#### 总结
>**状态机模型没有什么固定的模板，因为他就是一个思想，只需要把每一个点的状态表示全，并且每一个状态是由那几个状态转移来的。**
## 状态压缩dp
$~~~~$
**`状压dp简述`**
>**什么是状压dp ？**
>就是使用一个二进制来表示一个状态。
>**什么题使用状压dp ？**
>在一般的dp问题中，一般一个状态是可以由一个、两个或三个我们自己已知的状态转移过来的，但是一般在状压dp中一个状态的转移是不容易知道或不容易列举出来的，比如棋盘相关的问题，见下面题目。
>**状态压缩的优点 ？**
>一个状态可以由二进制表示，容易枚举，比如在背包问题中，101就表示的意思就是第1、3个物品选，第二个物品不选，而在存储状态的时候只需要存储101的十进制5就行了

#### 1.小国王（棋盘型状压dp）
![在这里插入图片描述](https://img-blog.csdnimg.cn/ef2013ddfbcb489d9e4e4133db46c4ba.png)
**解题思路**

**`拓展：“八相邻”：一个点的上下左右，左上右上左下右下八个方向相邻；另外在搜索问题中八相邻的枚举通常使用两层循环，而四相邻枚举通常采用偏移量`**

`状态表示：f[i][j][k]表示前i层放置了j个国王且第i层的状态时k的方案数`
`状态转移：f[i][j][k] += f[i-1][j-cnt[j]][k]也就是前i-1层放置了的国王数（总的国王数-第i层放置的国王数）且第i层状态为k时的方案数。`


**代码如下**
```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long LL;
const int N = 12,M = 1<<10,K=110;
int n,m;
LL f[N][K][M];//摆好了前n行且摆了k个国王且第i行的状态为m的方案数
int cnt[M];//状态中所含1的数量（二进制表示的情况下）
vector<int> state;//所有合法的状态
vector<int> head[M];//两个状态之间能够转移

//检查这个状态是否相邻为1
bool check(int state)
{
    for(int i=0;i<n;i++)
        if((state>>i&1)&&(state>>i+1&1))
            return false;
    return true;
}

//状态中1的数量
int count(int state)
{
    int res = 0;
    for(int i=0;i<n;i++) res+=state>>i&1;
    return res;
}

int main()
{
    cin>>n>>m;
    //先将所有合法状态找出
    for(int i=0;i<1<<n;i++)
        if(check(i))
        {
            state.push_back(i);
            cnt[i] = count(i);
        }
    //然后将所有两两状态之间可以转移的连接起来
    for(int i=0;i<state.size();i++)
        for(int j=0;j<state.size();j++)
        {
            int a = state[i],b = state[j];
            if((a&b)==0&&check(a|b))
                head[i].push_back(j);
        }
    //进行状态计算
    f[0][0][0] = 1;
    for(int i=1;i<=n+1;i++)
        for(int j = 0;j<=m;j++)
            for(int a = 0;a<state.size();a++)
                for(int b : head[a])
                {
                    int c = cnt[state[a]];
                    if(j>=c)
                        f[i][j][a] += f[i-1][j-c][b];
                }
    cout<<f[n+1][m][0]<<endl;
    return 0;
    
}
```

#### 2. 玉米田
![在这里插入图片描述](https://img-blog.csdnimg.cn/176e6d4f1c2f4eaba755704151af83cb.png)
**解题思路**
`状态表示：f[i][j]表示前i层已经摆好且第i层的状态是j的方案数`
`状态转移：和上题一样但没有k个国王那层`

***代码如下***
```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 14,M = 1<<12,mod = 1e8;
int f[N][M],g[N];
int n,m;
vector<int> state;
vector<int> head[M];

//检查相邻是否有相邻的1
bool check(int state)
{
    return !(state&(state>>1));
}

int main()
{
    cin>>n>>m;
    for(int i=1;i<=n;i++)
        for(int j=0;j<m;j++)
        {
            int x;
            cin>>x;
            g[i]+=!x<<j;
        }
     //将所有合法状态找出
    for(int i=0;i<1<<m;i++)
        if(check(i))
            state.push_back(i);
    //然后找出两两可以转移的状态
    for(int i=0;i<state.size();i++)
        for(int j=0;j<state.size();j++)
        {
            int a = state[i],b = state[j];
            if((a&b)==0) 
                head[i].push_back(j);
        }
    //进行状态计算
    f[0][0] = 1;//不摆也是一种方案
    for(int i=1;i<=n+1;i++)
        for(int a=0;a<state.size();a++)
            for(int b : head[a])
            {
                if(g[i]&state[a]) continue;
                f[i][a] = (f[i][a]+f[i-1][b])%mod;
            }
    cout<<f[n+1][0]<<endl;
    return 0;
    
}
```
#### 总结
`棋盘dp公共点：vector<int> state表示所有的合法状态，vector<int> head[M]表示所有两两之间可以转移的关系，经常先将所有的状态按照题目要求枚举到state数组中。`
`常用的函数：int count(int state)用来求这个状态中1的个数，bool check(int state)用来判断这个状态是否合理，判断依据：a&b==0,check(a|b)。`
## 树形dp模型
$~~~~$

#### 1.没有上司的舞会
![在这里插入图片描述](https://img-blog.csdnimg.cn/7585fe6a12b54453a339dd0aa156fe50.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1549f0868efd4d148247a5387d3860e9.png)

***题目分析***

>类似与状态机模型：0表示不选该状态，1表示选该状态。
>因此就有两种情况：
>1. 如果选了当前节点，那他的子节点就不能选。
>2. 如果不选当前节点，那么他所有儿子节点可以选可以不选


**解题思路**
`状态表示：f[i][2]表示第i个职员选还是不选。`
`状态转移：f[i][0] += max(f[u][0],f[u][1])//第i个职员不参加
					f[i][1] += f[u][0]//第i个职员参加;`


**代码如下**

```c
#include<bits/stdc++.h>
using namespace std;
const int N=6010;
int n;
int e[N],ne[N],h[N],idx;
int f[N][2];
int happy[N];
bool has_father[N];

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}

int dfs(int u)
{
    f[u][1]=happy[u];
    for(int i=h[u];i!=-1;i=ne[i])
    {
        int j=e[i];
        dfs(j);
        f[u][0]+=max(f[j][0],f[j][1]);
        f[u][1]+=f[j][0];
    }
}

int main()
{
    cin>>n;
    for(int i=1;i<=n;i++) cin>>happy[i];
    memset(h,-1,sizeof h);
    for(int i=0;i<n-1;i++)
    {
        int a,b;
        cin>>a>>b;
        has_father[a]=true;
        add(b,a);
    }
    int root=1;
    while(has_father[root]) root++;
    dfs(root);
    cout<<max(f[root][0],f[root][1])<<endl;
    return 0;
}
```

#### 2. 树的最长路径
![在这里插入图片描述](https://img-blog.csdnimg.cn/887dbd9bbbdf448cbb19f6ff03967ced.png)

***题目分析***

$~~$**参考**：[树的最长路径](https://www.acwing.com/solution/content/29832/)
> 由题目易知树中的最长路径必然经过根节点，也就是最高的结点，
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/f666b784b44b47c29973a089961ef304.png)
> 本题中f[x]表示x的子节点到x的最长距离，而非以x为顶点的最长路径，因为本题的路径是两条边，所以f[x]表示其中一条边的最大值，f[y]f[z]同理，两条边加起来即为以u为顶点的最长路径。![在这里插入图片描述](https://img-blog.csdnimg.cn/8ab2b36a1c814e27915ddfc4eb1d13e2.png)


**代码如下**
```c

#include<bits/stdc++.h>

using namespace std;

const int N=10010;

int h[N],w[2*N],ne[2*N],e[2*N],idx;
int ans;
//图的邻接表存储方法
void add(int a,int b,int c)
{
    w[idx]=c,e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}

int dfs(int u,int father)//father表示u的父节点,因为该图为无向图,并且迭代过程中不能回到父节点,所以要特殊标记.
{
    int dist=0;
    //d1d2为最长路径和最短路径
    int d1=0,d2=0;//注意:路径中可以只包含一个点（题目）
    //所以题目中的结果一定不为负数,负的路径由此可以忽略掉
    for(int i=h[u];i!=-1;i=ne[i])
    {
        int j=e[i];
        if(j==father) continue;//因为有可能遍历到父节点，所以直接跳过
        int d=dfs(j,u)+w[i];//求出路径的长度
        dist=max(dist,d);//求f[x]的最大值
        //d1,d2求出以该点为顶点的最长路径
        if(d>=d1) d2=d1,d1=d;//最长路径和次长路径
        else if(d>d2) d2=d;
    }

    ans=max(ans,d1+d2);

    return dist;//返回当前点的f[x];
}

int main()
{
    memset(h,-1,sizeof(h));
    int n;
    cin>>n;
    for(int i=1;i<n;i++)//n-1条边
    {
        int a,b,w;
        scanf("%d%d%d",&a,&b,&w);
        add(a,b,w),add(b,a,w);
    }
    dfs(1,-1);//让其从根节点开始遍历
	cout<<ans<<endl;
	return 0;
}

```

#### 3. 树的中心
![在这里插入图片描述](https://img-blog.csdnimg.cn/ae7af22e7d2244f98a7e57dcfa499f98.png)
***题目分析***

>题目概述：（暴力思想）遍历图中每一个点并且找出距离这个点最长的路径，然后找出所有最长路径里面的最小值。
>	**思考**：`怎么知道这个树中的一个结点的最远距离`
>   1. 从当前节点往下走，找到最远距离。
>   2. 从当前节点向上走到父节点，再由父节点出发并且不会到本结点。
>      $~~~~$
>
>**这种题型一般三个步骤：**
>   1. 指定一个根节点。
>   2. 一次dfs遍历，计算出当前节点的子节点对当前节点的贡献
>   3. 再一次dfs遍历，计算出父节点对当前节点的贡献。然后合并统计答案

**代码如下**

代码中变量表示含义：
>d1[u]：存下u节点向下走的最长路径的长度
>d2[u]：存下u节点向下走的第二长的路径的长度
>p1[u]：存下u节点向下走的最长路径是从哪一个节点下去的
>p2[u]：存下u节点向下走的第二长的路径是从哪一个节点走下去的
>up[u]：存下u节点向上走的最长路径的长度

```c
#include <bits/stdc++.h>
using namespace std;
const int N = 10010,M = 2*N,INF = 0x3f3f3f3f;

int idx,e[M],h[N],ne[M],w[M];
int d1[N],d2[N],p1[N],p2[N],up[N];
int n,ans;

void add(int a,int b,int c)
{
    e[idx] = b,w[idx] = c,ne[idx] = h[a],h[a] = idx++;
}

int dfs_d(int u,int f)
{
    d1[u] = d2[u] = -INF;
    
    for(int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if(j == f) continue;
        int d = dfs_d(j, u) + w[i];
        if(d >= d1[u]) 
        {
            d2[u] = d1[u],d1[u] = d;
            p2[u] = p1[u],p1[u] = j;
        }
        else if(d > d2[u]) 
        {
            d2[u] = d;
            p2[u] = j;
        }
    }
    
    if(d1[u] == -INF) d1[u] = d2[u] = 0;
    return d1[u];
    
}

int dfs_up(int u,int f)
{
    for(int i = h[u]; i!= -1;i = ne[i])
    {
        int j = e[i];
        if(j == f) continue;
        if(p1[u] == j) up[j] = max(up[u], d2[u]) + w[i];
        else up[j] = max(up[u],d1[u]) + w[i];
        dfs_up(j,u);
    }
}

int main()
{
    cin>>n;
    memset(h,-1,sizeof h);
    for(int i=0;i<n-1;i++)
    {
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c),add(b,a,c);
    }
    
    dfs_d(1,-1);
    dfs_up(1,-1);
    
    int res = INF;
    
    for(int i=1;i<=n;i++) res = min(res,max(d1[i],up[i]));
    cout<<res<<endl;
    
    return 0;
}

```

#### 4. 战略游戏
![在这里插入图片描述](https://img-blog.csdnimg.cn/d2d641e622644b66b7df33d80bf7d953.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/fbf611780c464e37a16ec476e65e5c61.png)

***题目分析***
>本题也是一种状态机模型，0表示不选这个节点，1表示可以选这个节点。
>所以就有两种方案：
> 1. 如果这个节点放置了哨兵，那么与他相连的节点可以放置也可以不放置哨兵。
> 2. 如果这个节点没有放置哨兵，那么与他相连的节点必须至少放置一个哨兵。
>
> **本题类似于`没有上司的舞会`** 

**解题思路**
`状态表示：f[i][2]表示第i个节点选或者不选所能求得的最小士兵数量。`
`状态转移可看代码`

**代码如下**

```c
#include<bits/stdc++.h>
using namespace std;
const int N = 1510;

int idx,e[N],ne[N],h[N];
int f[N][2];//表示第i个结点选或者不选所能求得的最小士兵数目
int n;
bool st[N];

void add(int a,int b)
{
    e[idx] = b,ne[idx] = h[a],h[a] = idx++;
}

void dfs(int u)//有向图不用判断是不是能走到父节点
{
    f[u][0] = 0;
    f[u][1] = 1;
    
    for(int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        dfs(j);
        f[u][0] += f[j][1];
        f[u][1] += min(f[j][1],f[j][0]);
    }
    
    
}

int main()
{
    while(scanf("%d",&n) == 1)
    {
        memset(h,-1,sizeof h);
        idx = 0;
        memset(st,0,sizeof st);
        
        for(int i=0;i<n;i++)
        {
            int id=0,cnt = 0;
            scanf("%d:(%d)",&id,&cnt);
            while(cnt--)
            {
                int ver=0;
                cin>>ver;
                add(id,ver);
                st[ver] = true;
            }
        }
        
        int root = 0;
        while(st[root]) root++;
        dfs(root);
        cout<<min(f[root][0],f[root][1])<<endl;
        
    }
    return 0;
}
```


#### 总结
>树形dp主要求在一个树中与最长/最短距离有关的问题。
>常用模板：
>```c
>//将给出的节点进行邻接表存储
>void add(int a,int b)
>{
>	e[idx]=b,ne[idx]=h[a],h[a]=idx++;
>}
>//dfs进行相关逻辑
>void dfs(int u)
>{
>	for(int i=h[u];~i;i=ne[i])
>		dfs()//相关逻辑
>}
>```

## 数位dp模型
$~~~~$
 **什么是数位dp**
 &emsp;  参考：[数位dp基本概念](https://www.acwing.com/solution/content/66855/)
>数位：把一个数字按照个、十、百、千等等一位一位地拆开，关注它每一位上的数字。如果拆的是 十进制数，那么每一位数字都是 0~9，其他进制可类比十进制。
>数位 DP：用来解决一类特定问题，这种问题比较好辨认，一般具有这几个特征：
>1. 要求统计满足一定条件的数的数量
>2. 这些条件经过转化后可以使用「数位」的思想去理解和判断
>3. 输入会提供一个数字区间（有时也只提供上界）来作为统计的限制
>4. 上界很大（比如 $10^9$），暴力枚举验证会超时

**数位dp的套路**


#### 1. 数的度量
![在这里插入图片描述](https://img-blog.csdnimg.cn/d8a84833621c49d49cd070597bb63ad6.png)

**代码如下**
```c
#include<bits/stdc++.h>
using namespace std;
const int N = 35; //位数
int f[N][N];// f[a][b]表示从a个数中选b个数的方案数，即组合数的值
int K, B; //K是能用的1的个数，B是B进制

//求组合数：预处理
void init(){
    for(int i=0; i< N ;i ++)
        for(int j =0; j<= i ;j++)
            if(!j) f[i][j] =1;
            else f[i][j] =f[i-1][j] +f[i-1][j-1];
}
 //求区间[0,n]中的 “满足条件的数” 的个数
 //“满足条件的数”是指：一个数的B进制表示，其中有K位是1、其他位全是0
int dp(int n){
    if(n == 0) return 0; //如果上界n是0，直接就是0种
    vector<int> nums; //存放n在B进制下的每一位
    //把n在B进制下的每一位单独拿出来
    while(n) nums.push_back( n% B) , n/= B;
    int res = 0;//答案：[0,n]中共有多少个合法的数
    //last在数位dp中存的是：右边分支往下走的时候保存前面的信息 
    //遍历当前位的时候，记录之前那些位已经占用多少个1，那么当前还能用的1的个数就是K-last
    int last = 0; 

    //从最高位开始遍历每一位
    for(int i = nums.size()-1; i>= 0; i--){

        int x = nums[i]; //取当前位上的数

        if(x>0){ //只有x>0的时候才可以讨论左右分支


            //当前位填0，从剩下的所有位（共有i位）中选K-last个数。
            //对应于：左分支中0的情况，合法
            res += f[i][ K -last];//i个数中选K-last个数的组合数是多少，选出来这些位填1，其他位填0


            if(x > 1){
                //当前位填1，从剩下的所有位（共有i位）中选K-last-1个数。
                //对应于：左分支中填1的情况，合法
               if(K - last -1 >= 0) res += f[i][K -last -1];//i个数中选K-last-1个数填1的组合数是多少
               //对应于：左分支中其他情况（填大于1的数）和此时右分支的情况（右侧此时也>1），不合法！！！
               //直接break。
                break;
            }

            //上面统计完了**左分支**的所有情况，和右分支大于1的情况，

            //这个else 是x==1，
            //对应于：右分支为1的情况，即限定值为1的情况，也就是左分支只能取0
            //此时的处理是，直接放到下一位来处理
            //只不过下一位可使用的1的个数会少1，体现在代码上是last+1

            else {
                last ++;
                //如果已经填的个数last > 需要填的个数K，不合法break
                if(last > K) break;
            }

        }
        //上面处理完了这棵树的**所有**左分支，就剩下最后一种右分支的情况
        // 也就是遍历到最后1位，在vector中就是下标为0的地方：i==0；
        // 并且最后1位取0，才算作一种情况res++。因为最后1位不为0的话，已经被上面的ifelse处理了。
        if(i==0 && last == K) res++; 
    }

    return res;
}

int main(){
    init();
    int l,r;
    cin >>  l >> r >> K >>B;
    cout<< dp(r) - dp(l-1) <<endl;  
}

```

#### 2. 数字游戏
![在这里插入图片描述](https://img-blog.csdnimg.cn/31ffa4a159714cd88bccd7f52434729e.png)
**代码如下**
```c
#include<bits/stdc++.h>

using namespace std;

const int N = 15;

int f[N][N];//f[i][j]表示i位数且最高位为j的不降序的方案数

void init()//dp过程
{
    for(int i=0;i<=9;i++) f[1][i]=1;
    //初始化,因为只有一位的方案数只有一个
    for(int i=2;i<N;i++)
     for(int j=0;j<=9;j++)
      for(int k=j;k<=9;k++)//状态划分
       f[i][j]+=f[i-1][k];
}

int dp(int n)
{ 
    if(!n) return 1;//n=0,只有0这一种方案
    //因为当n=0时,下面的while循环无法通过,所以要进行特判
    vector<int> num;
    while(n) num.push_back(n%10),n/=10;
    int ans=0;
    int lt=0;//保存上一位的最大值
    for(int i=num.size()-1;i>=0;i--)
    {
        int x=num[i];

        for(int j=lt;j<x;j++) //左边分支,因为要保持不降序,所以j>=lt
        ans+=f[i+1][j];

        if(lt>x) break;//如果上一位最大值大于x的话,不构成降序,所以右边分支结束
        lt=x;

        if(!i) ans++;//全部枚举完了也同样构成一种方案
    }

    return ans;
}

int main()
{
    int n,m;

    init();

    while(cin>>n>>m) printf("%d\n",dp(m)-dp(n-1));

    return 0;
}


```

#### 总结
`数位dp模板`

>		也就是以数位的思路将每个数分解开来
>	```c
>	
>	int dp(int n)
>	{
>	if (!n) return 1;
>	vector<int> nums;
>	while (n) nums.push_back(n % 10), n /= 10;
>	int res = 0;
>	int last = 0;
>	for (int i = nums.size() - 1; i >= 0; i -- ）
>	    >    int x = nums[i];

**......以下的慢慢更新**
## 插头dp模型
## dp的优化方法

**说明**：**以上所涉及题目均来自[acwing](www.acwing.com)**