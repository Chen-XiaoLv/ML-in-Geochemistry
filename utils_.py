from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig
import pyecharts.faker as F
from pyecharts.faker import Faker
import numpy as np
import pandas as pd


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


if __name__ == '__main__':
    x_data = ['Apple', 'Huawei', 'Xiaomi', 'Oppo', 'Vivo', 'Meizu']
    y_data_1 = [123, 153, 89, 107, 98, 23]
    y_data_2 = [32, 213, 60, 167, 142, 45]
    # l=DrawLine(x_data,[y_data_1,y_data_2],["系列1","系列2"],"题目","品牌","数量")
    l=DrawScatter(x_data,[y_data_1,y_data_2],["系列1","系列2"],"题目","品牌","数量").d
    l.render("./Res/test01.html")

    y_data = [
        [980, 930, 650, 760, 810, 1000, 1000, 960, 960, ],
        [920, 930, 650, 760, 310, 1000, 100, 960, 960, ],
        [680, 930, 650, 260, 810, 1400, 1000, 960, 960, ],
        [780, 930, 650, 760, 810, 1000, 600, 930, 960, ],
        [980, 630, 650, 760, 810, 1000, 1000, 960, 960, ],
    ]
    b=DrawBox(y_data)
    b.render("./Res/test02.html")

    s=DrawMulStack([b,l])
    s.render("./Res/test03.html")

    p=DrawPie(F.Faker.values(),F.Faker.choose())
    p.render("./Res/test04.html")

