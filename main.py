import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
%matplotlib inline

user_data = pd.read_csv(r'D:\ali_TC_competition\user_action.csv',sep=',',encoding='utf-8')
user_data.info()
user_data.head(10)
user_data.isnull().count() #查看有无空值
print('各字段唯一值情况：','\n')
print('用户ID（user_id）:',user_data['user_id'].nunique())
print('商品ID（item_id）:',user_data['item_id'].nunique())
print('商品类别（item_category）:',user_data['item_category'].nunique())
# 总数据量
print('\n')
print('总数据大小：',len(user_data))
user_data['date'] = user_data['time'].map(lambda x: x.split(' ')[0])
user_data['day'] = user_data['time'].map(lambda x: x.split(' ')[0].replace("2014-",""))
user_data['hour'] = user_data['time'].map(lambda x: x.split(' ')[1])
user_data.head()
#Dataform数据类型转换，便于后续分析操作
user_data['user_id'] = user_data['user_id'].astype('object')
user_data['item_id'] = user_data['item_id'].astype('object')
user_data['item_category'] = user_data['item_category'].astype('object')
user_data['date'] = pd.to_datetime(user_data['date'])
user_data['hour'] = user_data['hour'].astype('int64')
behavior_user_count = user_data.groupby('behavior_type')['user_id'].count()#查看行为类型的用户量
#按月/日统计浏览量及独立浏览量
#月统计浏览量--pv_count
user_data['date_month'] = user_data['time'].map(lambda x: x.split('-')[1])
pv_month = user_data.groupby('date_month')['user_id'].count()
pv_month = pv_month.reset_index()
pv_month = pv_month.rename(columns={'user_id':'pv_count'})
print('月统计浏览量：\t')
print(pv_month)
print('****'*20)
#日统计浏览量--pv_count
pv_day = user_data.groupby('day')['user_id'].count()
pv_day = pv_day.reset_index()
pv_day = pv_day.rename(columns={'user_id':'pv_count'})
print('日统计浏览量：\t')
print(pv_day.head())
#月统计独立浏览量--uv_count
uv_month = user_data.groupby('date_month')['user_id'].apply(lambda x: len(x.unique()))
uv_month = uv_month.reset_index()
uv_month = uv_month.rename(columns = {'user_id':'uv_count'})
print(uv_month)
print('****'*20)
#日统计浏览量--uv_count
uv_day = user_data.groupby('day')['user_id'].apply(lambda x: len(x.unique()))
uv_day = uv_day.reset_index()
uv_day = uv_day.rename(columns = {'user_id':'uv_count'})
print(uv_day.head(10))
print(pv_day.describe()) #流量描述信息
print(uv_day.describe())
# 统计每月每小时的pv访问量
pv_hour = user_data.groupby(['day','hour'])['user_id'].count()
pv_hour = pv_hour.reset_index()
pv_hour = pv_hour.rename(columns={'user_id':'pv_hour'})
print(pv_hour.describe())
# 统计每月每小时的uv访问量
uv_hour = user_data.groupby(['day','hour'])['user_id'].apply(lambda x: len(x.unique()))
uv_hour = uv_hour.reset_index()
uv_hour = uv_hour.rename(columns={'user_id':'uv_hour'})
print(uv_hour.head())
print('--'*12)
print(uv_hour.describe())
user_data_11 =user_data.loc[user_data['date_month']=='11']
pv_hour_11 = user_data_11.groupby('hour')['user_id'].count().reset_index().rename(columns={'user_id':'pv_hour_11'})
uv_hour_11 = user_data_11.groupby('hour')['user_id'].apply(lambda x:len(x.unique())).reset_index().rename(columns={'user_id':'uv_hour_11'})
user_data_12 =user_data.loc[user_data['date_month']=='12']
pv_hour_12 = user_data_12.groupby('hour')['user_id'].count().reset_index().rename(columns={'user_id':'pv_hour_12'})
uv_hour_12 = user_data_12.groupby('hour')['user_id'].apply(lambda x:len(x.unique())).reset_index().rename(columns={'user_id':'uv_hour_12'})
plt.figure(figsize=(12,6),dpi=75)
plt.subplots_adjust(wspace=0.35,hspace=0.35)
plt.subplot(2,2,1)
plt.plot(pv_hour_11['hour'],pv_hour_11['pv_hour_11'],color='green')
plt.xlabel('11月')
plt.ylabel('流量')
plt.title('11每小时流量 pv_hour')
plt.subplot(2,2,2)
plt.plot(uv_hour_11['hour'],uv_hour_11['uv_hour_11'],color='green')
plt.xlabel('11月')
plt.ylabel('流量')
plt.title('11月每小时独立流量 uv_hour')
plt.subplot(2,2,3)
plt.plot(pv_hour_12['hour'],pv_hour_12['pv_hour_12'],color='green')
plt.xlabel('12月')
plt.ylabel('流量')
plt.title('12每小时流量 pv_hour')
plt.subplot(2,2,4)
plt.plot(uv_hour_12['hour'],uv_hour_12['uv_hour_12'],color='green')
plt.xlabel('12月')
plt.ylabel('流量')
plt.title('12月每小时独立流量 uv_hour')
#新增用户分析及可视化
new_user=user_data.groupby('user_id')['day'].min()   #各用户活跃的最早日期
new_user_cnt=new_user.value_counts() #各日新增用户数
new_user_cnt={'day':new_user_cnt.index,'new_cnt':new_user_cnt.values}
new_user_cnt=pd.DataFrame(new_user_cnt)
print(new_user_cnt.head())
#可视化
plt.figure(figsize=(18,10))
plt.title('new_user_cnt',fontsize=16,fontweight='bold',ha='center',va='center')
plt.bar(new_user_cnt['day'],new_user_cnt['new_cnt'],color='green',width=0.7)
plt.xlabel('day',fontsize=14)
plt.ylabel('user_cnt',fontsize=14)
plt.tick_params(axis='y',labelsize=14)
plt.xticks(rotation=70, fontsize=14)
plt.tick_params(axis='y',labelsize=12)
for a,b in zip(np.arange(len(new_user_cnt)),new_user_cnt['new_cnt']):
    plt.text(a,b,b,fontsize=12,va='bottom',ha='center')

plt.show()
new_user=user_data.groupby('user_id')['date'].min()#刚开始日期
buy_data = user_data[user_data['behavior_type']==4][['user_id','date']].drop_duplicates()
buy_data=pd.merge(new_user,buy_data,how='left',on='user_id',suffixes=('_first',''))
buy_data['date_first'] = pd.to_datetime(buy_data['date_first'])
buy_data['date'] = pd.to_datetime(buy_data['date'])#最近购买日期
buy_data['days'] = buy_data['date']-buy_data['date_first']#天数
#留存数据（30天的）
#使用数据透视表
keep_data = pd.pivot_table(buy_data,values='user_id',index='date_first',columns='days',aggfunc=lambda x:len(x.unique()), fill_value='').reset_index()
keep_data
#将单元格改为数值格式，用于后续计算留存比例
keep_data = keep_data.applymap(lambda x:pd.to_numeric(x,errors='ignore'))
create_index = keep_data.columns
keep_data_rate= keep_data.iloc[:,[0,1]]
for i in range(2,31):#这里我们算到30日留存率
    rate= keep_data[create_index[i]]/keep_data[create_index[1]]
    keep_data_rate = pd.concat([keep_data_rate,rate],axis=1)
keep_data_rate
keep_data_rate.columns=['首次登录日期','新增用户数','次日留存率','3日留存率','4日留存率','5日留存率','6日留存率','7日留存率','8日留存率','9日留存率','10日留存率','11日留存率','12日留存率','13日留存率','14日留存率','15日留存率','16日留存率','17日留存率','18日留存率','19日留存率','20日留存率','21日留存率','22日留存率','23日留存率','24日留存率','25日留存率','26日留存率','27日留存率','28日留存率','29日留存率','30日留存率']
keep_data_rate
#整体行为转化率分析
behavior_user_count = user_data.groupby('behavior_type')['user_id'].count()#查看各类行为用户量
click_cnt, fav_add_cnt, pay_cnt =  behavior_user_count[1], behavior_user_count[2]+behavior_user_count[3], behavior_user_count[4]
print('收藏/点击转化率：'+str((100 * fav_add_cnt / click_cnt).round(2))+"%")
print('点击到购买转化率: '+str((100 * pay_cnt / click_cnt).round(2))+"%")
print('加购物车/收藏到购买转化率: '+str((100 * pay_cnt / fav_add_cnt).round(2))+"%")
#每日行为转化率分析
day_behavior_user_count = user_data.groupby(['day','behavior_type'])['user_id'].count()#查看行为类型的用户量
day_behavior_user_count=pd.DataFrame(day_behavior_user_count)
day_behavior_user_count=day_behavior_user_count.reset_index()
days_list = list(day_behavior_user_count['day'].drop_duplicates())
click_cnt_day=[]
fav_add_cnt_day=[]
pay_cnt_day=[]
for t in days_list:
    df=day_behavior_user_count[day_behavior_user_count['day']==t]
    click_cnt =df[df['behavior_type']==1]['user_id']
    fav_add_cnt_1 =df[df['behavior_type']==2]['user_id']
    fav_add_cnt_2 =df[df['behavior_type']==3]['user_id']
    pay_cnt =df[df['behavior_type']==4]['user_id']
    click_cnt_day.append(click_cnt.iloc[0].tolist())
    fav_add_cnt_day.append(fav_add_cnt_1.iloc[0].tolist()+fav_add_cnt_2.iloc[0].tolist())
    pay_cnt_day.append(pay_cnt.iloc[0].tolist())
fav_add_click_r = []
pay_click_r = []
pay_fav_add_r = []
for i in range(0,31):
    fav_add_click_rate = 100*(fav_add_cnt_day[i]/click_cnt_day[i])
    pay_click_rate = 100*(pay_cnt_day[i]/click_cnt_day[i])
    pay_fav_add_rate = 100*(pay_cnt_day[i]/fav_add_cnt_day[i])
    fav_add_click_r.append(fav_add_click_rate)
    pay_click_r.append(pay_click_rate)
    pay_fav_add_r.append(pay_fav_add_rate)
#可视化
plt.figure(figsize=(20,10))
barWidth = 0.3
r1 = np.arange(0,31)
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
plt.bar(r1,fav_add_click_r,width = 0.3,color ="b",label='收藏/点击转化率')#,"hotpink","#556B2F"
plt.bar(r2,pay_click_r,width = 0.3,color ="r",label='点击到购买转化率')
plt.bar(r3,pay_fav_add_r,width = 0.3,color ="g",label='加购物车/收藏到购买转化率')
plt.xticks([r + barWidth for r in range(0,31)], days_list)
plt.xticks(rotation=70, fontsize=14)
plt.ylabel('转化率 %')
plt.xlabel('日期')
plt.title('三种情形的转化率')
plt.legend()
plt.show()
#对商品类别和行为类别进行分类，并计算用户数
item_data = user_data.groupby(['item_category','behavior_type'])['user_id'].count()
item_data=item_data.reset_index()
item_data=item_data.rename(columns={'user_id':'behavior_cnt'})
item_data=pd.DataFrame(item_data)
#查看浏览量前20的商品类别
item_data_pv= item_data[item_data['behavior_type']==1].sort_values(by='behavior_cnt',ascending=False)
item_data_pv_1 = item_data_pv.head(20)
print(item_data_pv_1)
print('---'*15)
#查看购买量前20的商品类别
item_data_buy= item_data[item_data['behavior_type']==4].sort_values(by='behavior_cnt',ascending=False)
item_data_buy_4=item_data_buy.head(20)
print(item_data_buy_4)
item_data_pb_merge=pd.merge(item_data_pv_1,item_data_buy_4,how='inner',on='item_category')
print('---'*15)
item_data_pb_merge['conversion_rate']= (item_data_pb_merge['behavior_cnt_y']/item_data_pb_merge['behavior_cnt_x'])*100
item_data_pb_merge.sort_values('behavior_cnt_x',ascending=False)
#前20的流量与购买转化率商品的可视化
plt.figure(figsize=(12,18))
plt.subplots_adjust(hspace=0.3)
plt.subplot(2,1,1)
plt.title('TOP20 PV',fontsize=14)
plt.bar(np.arange(item_data_pv_1.shape[0]),item_data_pv_1['behavior_cnt'],width=0.3,color='green')
plt.xticks(np.arange(item_data_pv_1.shape[0]),np.array(item_data_pv_1['item_category']),fontsize=10)
plt.tick_params(axis='y',labelsize=10)
plt.ylabel('流量',fontsize=12)
plt.twinx()
plt.plot(np.arange(item_data_pb_merge.shape[0]),item_data_pb_merge['conversion_rate'],color='orange',linestyle='-',linewidth=0.75,marker='o',markersize=3)
plt.tick_params(axis='y',labelsize=10)
plt.ylabel('conversion_rate',fontsize=12)

plt.subplot(2,1,2)
plt.title('TOP20 BUY',fontsize=14)
plt.bar(np.arange(item_data_buy_4.shape[0]),item_data_buy_4['behavior_cnt'],width=0.3,color='green')
plt.xticks(np.arange(item_data_buy_4.shape[0]),np.array(item_data_buy_4['item_category']),fontsize=10)
plt.tick_params(axis='y',labelsize=10)    #设置坐标轴字体大小
plt.ylabel('buy_cnt',fontsize=12)
plt.twinx()
plt.plot(np.arange(item_data_pb_merge.shape[0]),item_data_pb_merge['conversion_rate'],color='orange',linestyle='-',linewidth=0.75,marker='o',markersize=3)
plt.tick_params(axis='y',labelsize=10)
plt.ylabel('conversion_rate',fontsize=12)
plt.show()
#统计用户购买频次
user_data_buy_freq = user_data[user_data.behavior_type==4].groupby('user_id')['behavior_type'].count()
user_data_buy_freq = user_data_buy_freq.reset_index()
user_data_buy_freq = user_data_buy_freq.rename(columns={'behavior_type':'buy_cnt'})
user_data_buy_freq.describe()
user_data_buy_freq.plot(x='user_id',y='buy_cnt',title="Purchase frequency",color='green')
data_day_num = user_data.groupby('day')['user_id'].apply(lambda x: len(x.unique()))
data_day_pay = user_data[user_data.behavior_type==4].groupby('day')['user_id'].count()
day_user_num = user_data[user_data.behavior_type==4].groupby('day')['user_id'].apply(lambda x: len(x.unique()))
ARPU = data_day_pay/data_day_num
ARPPU = data_day_pay/day_user_num
# 可视化
plt.figure(figsize=(16,10))
plt.subplots(1,2)
plt.subplot(1,2,1)
ARPU.plot(color='green')
plt.title('ARPU')
ARPU = data_day_pay/data_day_num
plt.subplot(1,2,2)
ARPPU.plot(color='green')
plt.title('ARPPU')
plt.show()
from normal_comparsion_plot.Norm_comparision_plot import norm_comparision_plot
#R:最近一次消费时间统计
recent_time=user_data[user_data.behavior_type==4]['date'].max()#获取最新日期
recent_buy_time=user_data[user_data.behavior_type==4].groupby('user_id')['date'].max().to_frame() #各用户最近一次购买日期
recent_buy_time['R']=(recent_time-recent_buy_time['date']).dt.days #每个用户最后一次购买时间距今多少天
print(recent_buy_time.head(15))
#F:统计每个用户累计消费频次。即一天内用户消费的次数，同一天多次消费也算一次
buy_freq=user_data[user_data.behavior_type==4].drop_duplicates(subset=('user_id','date')).groupby('user_id')['date'].count().to_frame().rename(columns={'date':'F'})
print(buy_freq.head(15))
#M:代表每个客户平均购买金额，也可以是累计购买金额,在这里我们用当天消费频率来表示累计购买金额，即每个用户当日总消费次数/消费次数
behavior_buy_num_day = user_data[user_data.behavior_type==4].drop_duplicates(subset=('user_id','date')).groupby('user_id')['date'].count().to_frame().rename(columns={'date':'buy_num'})
behavior_buy_cnt= user_data[user_data.behavior_type==4].groupby('user_id')['date'].count().to_frame().rename(columns={'date':'total_buy_num'})
behavior_buy_num=pd.merge(behavior_buy_cnt,behavior_buy_num_day,on='user_id')
behavior_buy_num['M']= behavior_buy_num['total_buy_num']/behavior_buy_num['buy_num']
select_cols=['M']
buy_mon=behavior_buy_num[select_cols]
print(buy_mon.head())
#合并R、F、M
RFM=pd.merge(recent_buy_time,buy_freq,left_index=True,right_index=True)
RFM=pd.merge(RFM,buy_mon,left_index=True,right_index=True)
#查看R,F,M的分布，进行合理的划分区间
norm_comparision_plot(RFM['R'],figsize=(12,6))
norm_comparision_plot(RFM['F'],figsize=(12,6))
norm_comparision_plot(RFM['M'],figsize=(12,6))
#结合分布图给RFM打分
#给R打分
R_bins = [0,4,8,12,16,1000000]
R_labels = [1,2,3,4,5]
RFM['R_score']=pd.cut(RFM['R'],bins=R_bins,labels=R_labels,right=False)
#给F打分
F_bins = [1,6,11,16,21,1000000]
F_labels = [5,4,3,2,1]
RFM['F_score']=pd.cut(RFM['F'],bins=F_bins,labels=F_labels,right=False)
#给M打分
M_bins = [0,2,4,6,8,1000000]
M_labels = [1,2,3,4,5]
RFM['M_score']=pd.cut(RFM['R'],bins=M_bins,labels=M_labels,right=False)
print(RFM.info())
#转换格式
RFM['R_score']=RFM['R_score'].astype(float)
RFM['F_score']=RFM['F_score'].astype(float)
RFM['M_score']=RFM['M_score'].astype(float)
RFM['R_above_mean']=(RFM['R_score']>RFM['R_score'].mean())*1
RFM['F_above_mean']=(RFM['F_score']>RFM['F_score'].mean())*1
RFM['M_above_mean']=(RFM['M_score']>RFM['M_score'].mean())*1
#RFM总分
RFM['total_score']=RFM['R_above_mean']*100+RFM['F_above_mean']*10+RFM['M_above_mean']*1
#用户分层
user_labels={111:'重要用户',110:'消费潜力用户',101:'频次深耕用户',100:'新用户',11:'重要价值流失预警用户',10:'一般用户',1:'高消费唤回用户',0:'流失用户'}
RFM['user_type']=RFM['total_score'].map(user_labels)
#RFM可视化
RFM_CNT=RFM['user_type'].value_counts().to_frame()
sizes=RFM_CNT
plt.figure(figsize=(12,6))
plt.axis('equal')
plt.title('RFM',fontsize=12)
plt.pie(RFM_CNT['user_type'],labels=RFM_CNT.index,
        colors=cm.GnBu(np.arange(len(sizes)) / len(sizes)),
        autopct='%1.2f%%', #标签百分比的显示格式
        pctdistance=0.75,  #数字标签距离圆心的距离
        textprops={'fontsize':10})
plt.show()
#定义RFM分布图函数及可视化
from scipy import stats
from scipy.stats import norm, skew
plt.style.use('fivethirtyeight')
cluster_data=RFM[['R','F','M']]
def draw_dist_prob(data):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(24, 12), dpi=300)

    for i,j in enumerate(['R', 'F', 'M']):
        sns.distplot(data[j], fit=norm, ax=ax[0][i])
        (mu, sigma) = norm.fit(data[j])
        ax[0][i].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
        ax[0][i].set_ylabel('数量')
        ax[0][i].set_title('{} 频数图'.format(j))

        stats.probplot(data[j], plot=ax[1][i])

draw_dist_prob(cluster_data)
#观察分布图可知R、F、M不服从正态分布，于是取开平方
cluster_data=RFM[['R','F','M']]
for i in cluster_data.columns:
    cluster_data[i]= cluster_data[i]+0.000001
    cluster_data[i]=np.sqrt(cluster_data[i])
draw_dist_prob(cluster_data)
#肘部法选择K值
from sklearn.cluster import KMeans
# 选择K的范围 ，遍历每个值进行评估
inertia_list=[]
for k in range(1,10):
    model=KMeans(n_clusters=k,max_iter=500,random_state=12)
    cluster_data_scaler_kmeans=model.fit(cluster_data_scaler)
    inertia_list.append(cluster_data_scaler_kmeans.inertia_)

# 绘图
fig,ax =plt.subplots(figsize=(12,6))
ax.plot(range(1,10),inertia_list,'o-',linewidth=1)
ax.set_xlabel('k',fontsize=12)
ax.set_ylabel("inertia_score",fontsize=12)
ax.set_title('inertia变化图',fontsize=12)
plt.show()
from sklearn import metrics
label_list=[]
silhouette_score_list=[]
for k in range(2,10):
    model=KMeans(n_clusters=k,max_iter=500,random_state=100)
    kmeans=model.fit(cluster_data_scaler)
    silhouette_score=metrics.silhouette_score(cluster_data_scaler,kmeans.labels_)  # 轮廓系数
    silhouette_score_list.append(silhouette_score)
    label_list.append({k:kmeans.labels_})

# 绘图
fig,ax=plt.subplots(figsize=(12,6))
ax.plot(range(2,10),silhouette_score_list,'o-',linewidth=1)
ax.set_xlabel('k',fontsize=12)
ax.set_ylabel("silhouette_score",fontsize=12)
ax.set_title('轮廓系数变化图',fontsize=12)
plt.show()
calinski_harabaz_score_list=[]
for i in range(2,10):
    model=KMeans(n_clusters=i,random_state=100)
    kmeans=model.fit(cluster_data_scaler)
    calinski_harabaz_score=metrics.calinski_harabasz_score(cluster_data_scaler,kmeans.labels_)
    calinski_harabaz_score_list.append(calinski_harabaz_score)

# 绘图
fig,ax = plt.subplots(figsize=(12,6))
ax.plot(range(2,10),calinski_harabaz_score_list,'o-',linewidth=1)
ax.set_xlabel('k',fontsize=12)
ax.set_ylabel("calinski_harabaz_score",fontsize=12)
ax.set_title('calinski_harabaz_score变化图',fontsize=12)
plt.show()
# 分为3类
model=KMeans(n_clusters=3,random_state=1000)
kmeans=model.fit(cluster_data_scaler)
cluster_data['label']=kmeans.labels_
cluster_data_scaler['label']=kmeans.labels_
print(cluster_data_scaler)
#可视化
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(48,12),dpi=200)

ax=plt.axes(projection='3d')
ax1=ax.scatter(cluster_data_scaler[cluster_data_scaler['label']== 0].R, cluster_data_scaler[cluster_data_scaler['label']== 0].F,cluster_data_scaler[cluster_data_scaler['label']== 0].M, edgecolor = 'k', color = 'r')
ax2=ax.scatter(cluster_data_scaler[cluster_data_scaler['label']==1].R, cluster_data_scaler[cluster_data_scaler['label']==1].F,cluster_data_scaler[cluster_data_scaler['label']==1].M, edgecolor = 'k', color = 'b')
ax3=ax.scatter(cluster_data_scaler[cluster_data_scaler['label']==2].R, cluster_data_scaler[cluster_data_scaler['label']==2].F,cluster_data_scaler[cluster_data_scaler['label']==2].M, edgecolor = 'k', color = 'y')
ax.legend([ax1,ax2,ax3],['Cluster 1','Cluster 2','Cluster 3'],fontsize=12)
ax.set_xlabel('R',fontsize=16)
ax.set_ylabel('F',fontsize=16)
ax.set_zlabel('M',fontsize=16)
ax.set_title('K-Means Clusters',fontsize=16)
plt.show()
#计算聚类后各类分组的数量与占比
cluster_data_scaler_cnt=cluster_data_scaler['label'].value_counts().to_frame()
sizes=cluster_data_scaler_cnt
cluster_cnt=cluster_data_scaler['label'].value_counts(ascending=False)
cluster_rate=cluster_data_scaler['label'].value_counts(normalize=True,ascending=False).apply(lambda x:format(x,'.2%'))
#计算各分组的平均数
cluster_mean=pd.pivot_table(cluster_data_scaler,index='label',values=['R','F','M'],aggfunc={'R':'mean','F':'mean','M':'mean'}).round(2)
#将以上数据合并
cluster_result=pd.concat([cluster_cnt,cluster_rate,cluster_mean],axis=1)
cluster_result.columns=['人数','人数占比','R均值','F均值','M均值']
cluster_result.index=['group_0','group_1','group_2']
print('基于RFM的聚类结果--\n{}'.format(cluster_result))
plt.figure(figsize=(12,6))
plt.axis('equal')
plt.title('K-Means Clusters Based on RFM',fontsize=12)
plt.pie(cluster_result['人数'],labels=cluster_result.index,
        colors=cm.GnBu(np.arange(len(sizes)) / len(sizes)),
        autopct='%1.1f%%', #标签百分比的显示格式
        pctdistance=0.75,  #数字标签距离圆心的距离
        textprops={'fontsize':10})
plt.show()
