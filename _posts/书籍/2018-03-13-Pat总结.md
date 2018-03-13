# PAT总结

## PAT中常用的STL容器

### 顺序容器

#### vector

```c++
#include<vector>
using namespace std;
vector<DataType> v;
v.push_back(Data);    //尾部添加元素
int size = v.size(); //元素个数
bool isEmpty = v.empty(); //是否为空
v.pop_back(); //删除末尾元素
v.clear(); //清空元素
//遍历
int length = v1.size();
for(int i=0;i<length;i++)
{
    cout<<v[i];
}
```

#### queue

```c++
#include<queue>
queue<Datatype> q;
q.push(Data); //入队
q.pop(); //队首出队
q.front(); //队首元素
q.back(); //队尾元素
q.empty(); //判断队列是否为空
q.size(); //队列元素个数
```

#### priority_queue

```c++
#include <iostream>
#include <queue>
using namespace std;
struct cmp{
    bool operator()(int a,int b){
    return a>b;
    }
};
/**************************************
struct Node{
    int a
};
bool operator<(Node a,Node b){
    return a>b;
}
**************************************/
int main(){
    priority_queue<int,vector<int>,cmp> q;
    //priority_queue<int,vector<int>,greater<int> > q;
    for(int i=0;i<10;i++){
        q.push(i);
    }
    while(!q.empty()){
        cout<<q.top()<<" ";
        q.pop();
    }
    cout<<endl;
    for(int i=9;i>=0;i--){
        q.push(i);
    }
    while(!q.empty()){
        cout<<q.top()<<" ";
        q.pop();
    }
    return 0;
}
```

#### stack

```c++
#include<stack>
stack<Datatype> s;
s.push(Data); //入栈
s.pop(); //出栈
s.top(); //访问栈顶元素
s.empty(); //判断栈是否为空
s.size(); //栈元素个数
```

### 关联容器

#### map

```c++
#include<map>
//1.定义和初始化
map<int,string> map1;                  //空map

//2.常用操作方法
map1[3] = "Saniya";                    //添加元素
map1.insert(map<int,string>::value_type(2,"Diyabi"));//插入元素
//map1.insert(pair<int,string>(1,"Siqinsini"));
map1.insert(make_pair<int,string>(4,"V5"));
string str = map1[3];                  //根据key取得value，key不能修改
map<int,string>::iterator iter_map = map1.begin();//取得迭代器首地址
int key = iter_map->first;             //取得key
string value = iter_map->second;       //取得value
map1.erase(iter_map);                  //删除迭代器数据
map1.erase(3);                         //根据key删除value
map1.size();                       //元素个数
map1.empty();                       //判断空
map1.clear();                      //清空所有元素

//3.遍历
for(map<int,string>::iterator iter = map1.begin();iter!=map1.end();iter++)
{
    int keyk = iter->first;
    string valuev = iter->second;
}
```

#### set

```c++
#include<map>
using namespace std;
set<int> s;
s.insert; //插入元素
s.erase(); //根据健值插入元素
s.clear(); //删除所有元素
s.empty(); //是否为空
s.find(); //返回指向查找元素位置的迭代器，找不到则返回s.end()
s.size(); //元素个数
//遍历
set<string>::iterator iter=set_str.begin();  
while(iter!=set_str.end())  
{  
    cout<<*iter<<endl;  
    ++iter;  
}  
```

## PAT常用算法

### 排序

#### STL函数

```c++
bool cmp(){}
sort(begin,end,cmp);
```

#### 冒泡排序

```c++
void BubbleSort(int A[], int n)
{
    for (int j = 0; j < n - 1; j++)         // 每次最大元素就像气泡一样"浮"到数组的最后
    {
        for (int i = 0; i < n - 1 - j; i++) // 依次比较相邻的两个元素,使较大的那个向后移
        {
            if (A[i] > A[i + 1])            // 如果条件改成A[i] >= A[i + 1],则变为不稳定的排序算法
            {
                swap(A[i], a[i+1]);
            }
        }
    }
}
```

#### 插入排序

```c++
void InsertionSort(int A[], int n)
{
    for (int i = 1; i < n; i++)         // 类似抓扑克牌排序
    {
        int get = A[i];                 // 右手抓到一张扑克牌
        int j = i;                  // 拿在左手上的牌总是排序好的
        while (j >0 && A[j-1] > get)    // 将抓到的牌与手牌从右向左进行比较
        {
            A[j] = A[j-1];            // 如果该手牌比抓到的牌大，就将其右移
            j--;
        }
        A[j] = get; // 直到该手牌比抓到的牌小(或二者相等)，将抓到的牌插入到该手牌右边(相等元素的相对次序未变，所以插入排序是稳定的)
    }
}
```

#### 希尔排序

```c++
#include <stdio.h>  

// 分类 -------------- 内部比较排序
// 数据结构 ---------- 数组
// 最差时间复杂度 ---- 根据步长序列的不同而不同。已知最好的为O(n(logn)^2)
// 最优时间复杂度 ---- O(n)
// 平均时间复杂度 ---- 根据步长序列的不同而不同。
// 所需辅助空间 ------ O(1)
// 稳定性 ------------ 不稳定

void ShellSort(int A[], int n)
{
    int h = 0;
    while (h <= n)                          // 生成初始增量
    {
        h = 3 * h + 1;
    }
    while (h >= 1)
    {
        for (int i = h; i < n; i++)
        {
            int j = i - h;
            int get = A[i];
            while (j >= 0 && A[j] > get)
            {
                A[j + h] = A[j];
                j = j - h;
            }
            A[j + h] = get;
        }
        h = (h - 1) / 3;                    // 递减增量
    }
}
```

#### 归并排序

```c++
void Merge(int A[], int left, int mid, int right)// 合并两个已排好序的数组A[left...mid]和A[mid+1...right]
{
    int len = right - left + 1;
    int *temp = new int[len];       // 辅助空间O(n)
    int index = 0;
    int i = left;                   // 前一数组的起始元素
    int j = mid + 1;                // 后一数组的起始元素
    while (i <= mid && j <= right)
    {
        temp[index++] = A[i] <= A[j] ? A[i++] : A[j++];  // 带等号保证归并排序的稳定性
    }
    while (i <= mid)
    {
        temp[index++] = A[i++];
    }
    while (j <= right)
    {
        temp[index++] = A[j++];
    }
    for (int k = 0; k < len; k++)
    {
        A[left++] = temp[k];
    }
}

void MergeSortRecursion(int A[], int left, int right)    // 递归实现的归并排序(自顶向下)
{
    if (left == right)    // 当待排序的序列长度为1时，递归开始回溯，进行merge操作
        return;
    int mid = (left + right) / 2;
    MergeSortRecursion(A, left, mid);
    MergeSortRecursion(A, mid + 1, right);
    Merge(A, left, mid, right);
}
```

#### 堆排序

```c++
// 分类 -------------- 内部比较排序
// 数据结构 ---------- 数组
// 最差时间复杂度 ---- O(nlogn)
// 最优时间复杂度 ---- O(nlogn)
// 平均时间复杂度 ---- O(nlogn)
// 所需辅助空间 ------ O(1)
// 稳定性 ------------ 不稳定
priority_queue<int> q;
void HeapSort(int A[], int n)
{
    int heap_size = BuildHeap(A, n);    // 建立一个最大堆
    while (q.size() > 0)    　　　　　　 // 堆（无序区）元素个数大于1，未完成排序
    {
        cout<<q.top();
        q.pop();
    }
}
```

#### 快速排序

```c++
#include <stdio.h>
int a[101],n;//定义全局变量，这两个变量需要在子函数中使用
void quicksort(int left,int right)
{
    int i,j,t,temp;
    if(left>right)
       return;
    temp=a[left]; //temp中存的就是基准数
    i=left;
    j=right;
    while(i!=j)
    {
        //顺序很重要，要先从右边开始找
        while(a[j]>=temp && i<j)
                j--;
        //再找右边的
        while(a[i]<=temp && i<j)
                i++;
        //交换两个数在数组中的位置
        if(i<j)
        {
                t=a[i];
                a[i]=a[j];
                a[j]=t;
        }
    }
    //最终将基准数归位
    a[left]=a[i];
    a[i]=temp;
    quicksort(left,i-1);//继续处理左边的，这里是一个递归的过程 
    quicksort(i+1,right);//继续处理右边的 ，这里是一个递归的过程 
}
```

#### 拓扑排序

```c++
#include<iostream>
#include <list>
#include <queue>
using namespace std;

/************************类声明************************/
class Graph
{
    int V;             // 顶点个数
    list<int> *adj;    // 邻接表
    queue<int> q;      // 维护一个入度为0的顶点的集合
    int* indegree;     // 记录每个顶点的入度
public:
    Graph(int V);                   // 构造函数
    ~Graph();                       // 析构函数
    void addEdge(int v, int w);     // 添加边
    bool topological_sort();        // 拓扑排序
};

/************************类定义************************/
Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];

    indegree = new int[V];  // 入度全部初始化为0
    for(int i=0; i<V; ++i)
        indegree[i] = 0;
}

Graph::~Graph()
{
    delete [] adj;
    delete [] indegree;
}

void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w);
    ++indegree[w];
}

bool Graph::topological_sort()
{
    for(int i=0; i<V; ++i)
        if(indegree[i] == 0)
            q.push(i);         // 将所有入度为0的顶点入队

    int count = 0;             // 计数，记录当前已经输出的顶点数 
    while(!q.empty())
    {
        int v = q.front();      // 从队列中取出一个顶点
        q.pop();

        cout << v << " ";      // 输出该顶点
        ++count;
        // 将所有v指向的顶点的入度减1，并将入度减为0的顶点入栈
        list<int>::iterator beg = adj[v].begin();
        for( ; beg!=adj[v].end(); ++beg)
            if(!(--indegree[*beg]))
                q.push(*beg);   // 若入度为0，则入栈
    }

    if(count < V)
        return false;           // 没有输出全部顶点，有向图中有回路
    else
        return true;            // 拓扑排序成功
}
```

### 最短路径

#### Dijkstra

```c++
const int inf = 99999999;
const int size = 100;
vector<int> pre[size];
int dis[size];
int e[size][size];
int root = 0;
bool visit[size];
fill(e[0],e[0]+size*size,inf);
fill(dis,dis+size,inf);
visit[root] = true;
dis[root] = 0;
void Dijkstra(){
    for(int i=0;i<size;i++){
        int u=-1,minn=inf;
        for(int j=0;j<size;j++){
            if(!visit[j] && dis[j]<minn){
                minn = dis[j];
                u = j;
            }
        }
        if(u=-1) return;
        visit[u] = true;
        for(int v=0;v<size;v++){
            if(!visit[v]&&e[u][v]!=inf){
                if(e[u][v]+dis[u]<dis[v]){
                    dis[v] = dis[u]+e[u][v];
                    pre[v].clear();
                    pre[v].push_back();
                }else if(e[u][v]+dis[u]==dis[v]){
                    pre[v].push_back(u);
                }
            }
        }
    }
}
```

#### Floyd算法

```c++
typedef struct
{
    char vertex[VertexNum];                                //顶点表
    int edges[VertexNum][VertexNum];                       //邻接矩阵,可看做边表
    int n,e;                                               //图中当前的顶点数和边数
}MGraph;

void Floyd(MGraph g)
{
 　　int A[MAXV][MAXV];
 　　int path[MAXV][MAXV];
 　　int i,j,k,n=g.n;
 　　for(i=0;i<n;i++){
    　　for(j=0;j<n;j++)
    　　{
             A[i][j]=g.edges[i][j];
         　　 path[i][j]=-1;
     　 }
    }
 　　for(k=0;k<n;k++){
      　　for(i=0;i<n;i++){
         　　for(j=0;j<n;j++){
             　　if(A[i][j]>(A[i][k]+A[k][j])){
                   　　A[i][j]=A[i][k]+A[k][j];
                   　　path[i][j]=k;
              　}
            }
        }
    }
}
```

### 图和树的遍历

#### DFS

```c++
vector<int> tree[size];
vector<int> temppath,path;
void dfs(int root){
    if(tree[root].size()==0){
        temppath.push_back(root);
        path = temppath;
        temp.pop_back();
        return;
    }
    temppath.push_back(root);
    for(int i=0;i<tree[root].size();i++){
        dfs(tree[root][i]);
    }
    temppath.pop_back();
}
```

#### BFS

```c++
queue<int> q;
vector<int> tree[size];
bool visit[size];
void bfs(){
    q.push_back(root);
    visit[root] = true;
    int first=0,last=1,depth=0,cnt=1;
    while(!q.empty()){
        int temp = q.front();
        q.pop();
        first++;
        for(int i=0;i<tree[temp].size();i++){
            if(!visit[tree[temp][i]]){
                q.push(tree[temp][i]);
                visit[tree[temp][i]] = true;
                cnt++;
            }
        }
        if(first==last){
            last = cnt;
            dep++;
        }
    }
}

```

### 树的前中后序遍历

#### 已知后序与中序输出前序

```c+
#include <cstdio>
using namespace std;
int post[size];
int in[size];
void pre(int root, int start, int end) {
    if(start > end) return ;
    int i = start;
    while(i < end && in[i] != post[root]) i++;
    printf("%d ", post[root]);
    pre(root - 1 - end + i, start, i - 1);
    pre(root - 1, i + 1, end);
}

int main() {
    pre(size, 0, size);
    return 0;
}
```

#### 已知后序与中序输出层序

```c++
#include <cstdio>
#include <vector>
using namespace std;
vector<int> post, in, level(100000, -1);
void pre(int root, int start, int end, int index) {
    if(start > end) return ;
    int i = start;
    while(i < end && in[i] != post[root]) i++;
    level[index] = post[root];
    pre(root - 1 - end + i, start, i - 1, 2 * index + 1);
    pre(root - 1, i + 1, end, 2 * index + 2);
}
```

#### 前序后序转中序

```c++
#include <cstdio>
#include <vector>
using namespace std;
vector<int> ans;
int *pre, *post, unique = 1;
int findFromPre (int x, int l, int r) {
    for (int i = l; i <= r; i++) {
        if (x == pre[i]) {
            return i;
        }
    }
    return -1;
}
void setIn (int prel, int prer, int postl, int postr) {
    if (prel == prer) {
        ans.push_back(pre[prel]);
        return;
    }
    if (pre[prel] == post[postr]) {
        int x = findFromPre(post[postr - 1], prel + 1, prer);
        if (x - prel > 1) {
            setIn(prel + 1, x - 1, postl, postl + x - prel - 2);
            ans.push_back(post[postr]);
            setIn(x, prer, postl + x - prel - 2 + 1, postr - 1);
        } else {
            unique = 0;
            ans.push_back(post[postr]);
            setIn(x, prer, postl + x - prel - 2 + 1, postr - 1);
        }
    }
}
```

### AVL树

```c++
struct Node{
    int val;
    struct Node *left,*right;
};
struct Node* leftRotate(struct Node *tree) {
    struct Node *temp = tree->right;
    tree->right = temp->left;
    temp->left = tree;
    return temp;
}
struct Node* rightRotate(struct Node *tree) {
    struct Node *temp = tree->left;
    tree->left = temp->right;
    temp->right = tree;
    return temp;
}
int getHeight(struct Node *tree) {
    if (tree == NULL) {
        return 0;
    } else {
        int l = getHeight(tree->left);
        int r = getHeight(tree->right);
        return l > r ? l + 1 : r + 1;
    }
}
struct Node* leftRightRotate(struct Node *tree) {
    tree->left = leftRotate(tree->left);
    tree = rightRotate(tree);
    return tree;
}
struct Node* rightLeftRotate(struct Node *tree) {
    tree->right = rightRotate(tree->right);
    tree = leftRotate(tree);
    return tree;
}
struct Node* insert(struct Node *tree, int val) {
    if (tree == NULL) {
        tree = new struct Node();
        tree->val = val;
        return tree;
    }
    if (tree->val > val) {
        tree->left = insert(tree->left, val);
        int l = getHeight(tree->left);
        int r = getHeight(tree->right);
        if (l - r >= 2) {
            if (val < tree->left->val) {
                tree = rightRotate(tree);
            } else {
                tree = leftRightRotate(tree);
            }
        }
    } else {
        tree->right = insert(tree->right, val);
        int l = getHeight(tree->left);
        int r = getHeight(tree->right);
        if (r - l >= 2) {
            if (val > tree->right->val) {
                tree = leftRotate(tree);
            } else {
                tree = rightLeftRotate(tree);
            }
        }
    }
    return tree;
}
```

### 并查集

```c++
int father[10010];
int visit[10010];
int find(int a){
    while(a!=father[a]){
        father[a] = father[father[a]];
        a = father[a];
    }
    return a;
}
void Union(int a,int b){
    int fa = find(a);
    int fb = find(b);
    if(fa!=fb){
        father[fa] = fb;
    }
}
```

### 树状数组

```c++
#define lowbit(i)((i)&(-i))
void update(int x, int v) {
    for(int i = x; i < maxn; i += lowbit(i))
        c[i] += v;
}
int getsum(int x) {
    int sum = 0;
    for(int i = x; i >= 1; i -= lowbit(i))
        sum += c[i];
    return sum;
}
```

### 背包问题

#### 01背包问题

```c+
int w[10010], dp[10010], v[10010];
bool choice[10010][10010];
int cmp1(int a, int b){return a > b;}
int main(){
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        cin>>w[i];
        cin>>v[i];
    }
    for(int i=1;i<=n;i++){
        for(int j=m;j>=w[i];j--){
            if(dp[j]<=dp[j-w[i]]+v[i]){
                dp[j] = dp[j-w[i]]+v[i];
            }
        }
    }
    return 0;
}
```

#### 完全背包

将01背包内循环的逆序改为顺序

### 数学问题

#### gcd

```c++
int gcd(int a, int b){ return b == 0 ? a : gcd(b,a%b); }
```

#### lcm

```c++
int lcm(ina ,int b){return a*b/gcd(a,b);}
```

#### 素数表的建立

```c++
bool prime[size] = {true};
for(int i=0;i*i<size;i++){
    for(int j=0;j*i<size;j++){
        prime[j*i] = false;
    }
}
```

### 字符串处理

```c++
string a = "123456";
int b = stoi(a); //convert int to string
string c = to_string(b); //convert string to int
sscanf(a,"%d",b); //从字符串a读进int类型的数据并放入b
sprintf(c,"%05d",b); //将b按格式化写入c
```