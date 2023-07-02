from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig
import pyecharts.faker as F
from pyecharts.faker import Faker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchinfo import summary

# 全局设置
CurrentConfig.ONLINE_HOST = "https://cdn.kesci.com/lib/pyecharts_assets/"

class Draw(object):
    def __init__(self,x,y,ylabel,title,xname,yname):
        self.x=x
        self.y=y
        self.ylabel=ylabel
        self.title=title
        self.xname=xname
        self.yname=yname

    def draw(self,tool):
        tool.add_xaxis(self.x)
        for i, j in enumerate(self.y):
            tool.add_yaxis(self.ylabel[i], j)
            # 这个是对每一组数据做的
            #
            # line.add_yaxis(ylabel[i],j,markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_='average',name='平均值')]
            #                                             ,linestyle_opts=opts.LineStyleOpts(type_='dashed' #点状
            #                                                                                ,opacity=0.9 #透明度 0-1 值越大越不透明
            #                                                                                ,color='black'
            #                                                                                )))
        # 设置最大最小标记
        tool.set_series_opts(markpoint_opts=opts.MarkPointOpts(
            data=[opts.MarkPointItem(type_="max", name="最大值"),  ##设置最大值 标记
                  opts.MarkPointItem(type_="min", name="最小值"),  # 设置最小值标记
                  ], symbol='pin', symbol_size=45), markline_opts=opts.MarkLineOpts(
            data=[opts.MarkLineItem(type_="average", name="平均值")]
        ))
        # 标记的图形。
        # ECharts 提供的标记类型包括 'circle', 'rect', 'roundRect', 'triangle',
        # 'diamond', 'pin', 'arrow', 'none'
        # 可以通过 'image://url' 设置为图片，其中 URL 为图片的链接，或者 dataURI。

        # 设置平均线

        # 设置工具箱
        tool.set_global_opts(title_opts=opts.TitleOpts(title='工具栏显示')
                             , toolbox_opts=opts.ToolboxOpts()
                             )
        return tool

# 折线图
class DrawLine(Draw):
    def __init__(self,x,y,ylabel,title,xname,yname):
        super(DrawLine, self).__init__(x,y,ylabel,title,xname,yname)
        tool=Line()
        self.d=self.draw(tool)
    def render(self,path):
        self.d.render(path)

def DrawHist(y, yname, title, bins=None):
    if bins == None:
        bins = np.linspace(min(y), max(y), 15)
    y = pd.cut(y, bins)
    value = pd.value_counts(y).sort_index()
    x_value = [f'{round(i.left, 2)}-{round(i.right, 2)}' for i in value.index]

    fit=np.arange(1,len(x_value)+1)
    p1=np.polyfit(fit,value,11)
    n = list(p1)[::-1]

    y1 = np.array([n[0]] * len(x_value))
    for i in range(1, len(n)):
        y1 += x_value ** i * n[i]

    value = value.tolist()
    c = (
        Bar()
            .add_xaxis(x_value)
            .add_yaxis(yname, value, category_gap=0, color=Faker.rand_color())
            .set_global_opts(title_opts=opts.TitleOpts(title=title))
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    x=DrawMulStack([c,DrawLine(fit,[y1],["n"],"n","n","n")])
    return x

# 条形图
class DrawBar(Draw):
    def __init__(self,x,y,ylabel,title,xname,yname):
        super(DrawBar, self).__init__(x,y,ylabel,title,xname,yname)
        tool=Bar()
        self.d=self.draw(tool)

    def render(self,path):
        self.d.render(path)



# 散点图
class DrawScatter(Draw):
    def __init__(self, x, y, ylabel, title, xname, yname):
        super(DrawScatter, self).__init__(x, y, ylabel, title, xname, yname)
        tool = Scatter()
        self.d = self.draw(tool)
    def render(self, path):
        self.d.render(path)

# 箱线图
def DrawBox(x,y_data):
    box_plot = Boxplot()
    box_plot = (
        box_plot.add_xaxis(xaxis_data=[i for i in x])
            .add_yaxis(series_name="箱线图", y_axis=box_plot.prepare_data(y_data))
            .set_global_opts(
            title_opts=opts.TitleOpts(
                pos_left="left", title="居左的标题"
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                boundary_gap=True,
                splitline_opts=opts.SplitLineOpts(is_show=False),  # 分割线显示与否
            ),
            yaxis_opts=opts.AxisOpts(  # y轴
                type_="value",
                name="km/s minus 299,000",
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)  # 横向分割
                ),
            ),
        )
            .set_series_opts(tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}"))  # 按照名称/最小值/Q1/中值/Q3/最大值 展示
    )
    return box_plot

# 多图叠加
def DrawMulStack(charts):
    grid = (
        Grid(init_opts=opts.InitOpts(width="1000px", height="600px"))  # 设置长宽
            # 箱线图的位置 调整数值之后可使得两者不交叠,可以不输入
            .add(
            charts[0],
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_bottom="15%"),
        )
            # 点点的位置 调整数值之后可使得两者不交叠，可以不输入
            .add(
            charts[1],
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_bottom="15%"),
        )
    )
    return grid

# 饼图
def DrawPie(x,y):
    pie = Pie()
    pie.add("", [list(z) for z in zip(y,x)],
            label_opts=opts.LabelOpts(position="left", formatter="{d}%"), radius=["30%", "50%"], rosetype="area" * 10)
    pie.set_global_opts(
        title_opts=opts.TitleOpts(title="饼图", pos_left="40%"),
        legend_opts=opts.LegendOpts(
            type_="scroll",
            pos_top="20%",
            pos_left="80%",
            orient="vertical"
        )
    )
    return pie




class ResNetBasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(ResNetBasicBlock, self).__init__()
        # ResNet
        # [[Conv2d(3x3)->BN->ReLU]*2-ReLU]+x]->ReLU
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1)
        # size: (w-k+1)/s
        self.bn1=nn.BatchNorm2d(out_channel)
        self.re=nn.ReLU()

        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(out_channel)

    def forward(self,x):
        out=self.re(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        return self.re(out+x)

class ResNetDownBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride[0],padding=1)
        self.bn1=nn.BatchNorm2d(out_channel)

        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride[1],padding=1)
        self.bn2=nn.BatchNorm2d(out_channel)

        self.extra=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride[0],padding=0),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self,x):
        extra_x=self.extra(x)
        out=self.conv1(x)
        out=F.relu(self.bn1(out))

        out=self.conv2(out)
        out=self.bn2(out)

        return F.relu(extra_x+out)


class BasicConv2d(nn.Module):
    # 一个Conv+Bn+ReLU
    def __init__(self,in_channel,out_channel,kernel,stride,padding=0):
        super(BasicConv2d, self).__init__()
        self.cbr=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,
                      padding=padding,bias=False),
            nn.BatchNorm2d(out_channel,eps=0.001,momentum=0.1,affine=True),
            nn.ReLU(inplace=False)
        )
    def forward(self,x):
        return self.cbr(x)

# 一个简单的Inception模块
class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()
        # Outsize
        # W= [W+2*padding-kernel]/stride+1
        # H= [H+2*padding-kernel]/stride+1

        # Why can cat?
        # cause when padding*2-kernel+1=0, they have the same width and height.
        # so: K3 p1 s1
        # k1 p0 s1
        # k5 p2 s1
        # they are the same
        self.branch0=BasicConv2d(192,96,kernel=1,stride=1)
        self.branch1=nn.Sequential(
            BasicConv2d(192,48,kernel=1,stride=1),
            BasicConv2d(48,64,5,1,2)
        )
        self.branch2=nn.Sequential(
            BasicConv2d(192,64,kernel=1,stride=1),
            BasicConv2d(64,96,kernel=3,stride=1,padding=1),
            BasicConv2d(96,96,kernel=3,stride=1,padding=1)
        )
        self.branch3=nn.Sequential(
            nn.AvgPool2d(3,stride=1,padding=1,count_include_pad=False),
            BasicConv2d(192,64,1,1)
        )

    def forward(self,x):
        x0=self.branch0(x)
        x1=self.branch1(x)
        x2=self.branch2(x)
        x3=self.branch3(x)
        out=torch.cat((x0,x1,x2,x3),1)
        return out

class Block35(nn.Module):
    def __init__(self,scale=1.0):
        super(Block35,self).__init__()
        self.scale=scale
        self.branch0=BasicConv2d(320,32,kernel=1,stride=1)
        self.branch1=nn.Sequential(
            BasicConv2d(320,32,1,1),
            BasicConv2d(32,32,3,1,1)
        )
        self.branch2=nn.Sequential(
            BasicConv2d(320,32,1,1),
            BasicConv2d(32,48,3,1,1),
            BasicConv2d(48,64,3,1,1)
        )
        self.conv2d=nn.Conv2d(128,320,kernel_size=1,stride=1)
        self.relu=nn.ReLU(inplace=False)

    def forward(self,x):
        x0,x1,x2=self.branch0(x),self.branch1(x),self.branch2(x)
        out=torch.cat((x0,x1,x2),1)
        out=self.conv2d(out)
        out=out*self.scale+x
        out=self.relu(out)
        return out

# ConvBlock
class Conv(nn.Module):
    # 这个就是最基本的卷积结构，为了方便shortcut,留下了激活函数接口
    def __init__(self,in_c,out_c,k_s=1,stride=1,padding=None,groups=1,activation=True):
        super(Conv, self).__init__()
        # padding这个参数，选择k_s//2就是为了让特征图的尺寸不发生改变
        padding=k_s//2 if padding is None else padding
        self.hidden=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=k_s,stride=stride,padding=padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act=nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self,x):
        return self.act(self.hidden(x))

# BasicBlock
class ResBlock(nn.Module):
    # 这是一个残差链接模块
    def __init__(self,in_c,out_c,down_sample=False,groups=1):
        super(ResBlock, self).__init__()
        # 通过过stride进行降采样，当然也不是每次都做
        stride=2 if down_sample else 1
        mid_channels=out_c//4 # 中间无所谓了

        # 如果做下采样的话，shortcut的大小也需要改变，这里就直接拿出一个不做ReLu的卷积来实现了
        self.shortcut=Conv(in_c,out_c,k_s=1,stride=1,activation=False) \
            if in_c!=out_c else nn.Identity()

        # 经典 1 3 1 深层网络
        self.conv=nn.Sequential(
            *[
                Conv(in_c,mid_channels,k_s=1,stride=1),
                Conv(mid_channels,mid_channels,k_s=3,stride=stride,groups=groups),
                Conv(mid_channels,out_c,k_s=1,stride=1,activation=False)
            ]
        )

    def forward(self,x):
        return F.relu(self.conv(x)+self.shortcut(x),inplace=True)

# ResNet50
class ResNet50(nn.Module):
    def __init__(self,num_classes):
        super(ResNet50, self).__init__()
        # Stem阶段
        self.stem=nn.Sequential(*[
            Conv(3,64,k_s=7,stride=2),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        ])

        # Net50
        self.stages=nn.Sequential(*[
            self._make_stage(64,256,down_sample=False,num_classes=3),
            self._make_stage(256,512,down_sample=True,num_classes=4),
            self._make_stage(512,1024,down_sample=True,num_classes=6),
            self._make_stage(1024,2048,down_sample=True,num_classes=3),
        ])

        # 输出阶段
        self.head=nn.Sequential(*[
            nn.AvgPool2d(kernel_size=7,stride=1,padding=0),
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(2048,num_classes)
        ])

    @staticmethod
    def _make_stage(in_channel,out_channel,down_sample,num_blocks):
        layers=[ResBlock(in_channel,out_channel,down_sample=down_sample)]
        for _ in range(1,num_blocks):
            layers.append(ResBlock(out_channel,out_channel,down_sample=False))
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.stages(self.stem(x))
        return self.head(x)



class NN(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(NN, self).__init__()
        self.hidden=nn.Sequential(
            nn.Linear(in_channel,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,7)
        )
    def forward(self,x):
        return self.hidden(x)
