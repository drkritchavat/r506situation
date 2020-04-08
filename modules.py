from plotly.subplots import make_subplots
import plotly
import ployly.graph_objects as go

def casebyprov(prov,dfwithmedian):
    nationmedian = dfwithmedian.loc[prov]
    nationmedian.index = nationmedian.index.astype('int')
    return nationmedian
def plotweek(nationmedian,visible):
    fig = [
        go.Scatter(x=nationmedian.index,y=nationmedian[2019],name='2019',marker_color='green',visible=visible),
        go.Scatter(x=nationmedian.index,y=nationmedian['median'],name='median',marker_color='red',visible=visible),
        go.Bar(x=nationmedian.index,y=nationmedian[2020],name='2020',marker_color='blue',visible=visible)
    ]
    return fig
def go_to_div(fig):
    div = plotly.offline.plot(fig,
                      include_plotlyjs=False,
                      output_type='div')
    return div

