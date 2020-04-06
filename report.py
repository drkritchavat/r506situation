import pandas as pd
import numpy as np
import mydb
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
import plotly
import pymongo
import plotly.express as px
from functools import reduce
import sys
from thai_strftime import thai_strftime
from bs4 import BeautifulSoup
from modules import casebyprov,plotweek,go_to_div

ds = sys.argv[1]
lastdate = sys.argv[2]
dlast = lastdate

with open('pquery.txt','r') as f:
    pquery = f.read()
pquery = pquery.replace('{ds}',ds)
with open('popquery.txt','r') as f:
    popquery = f.read()

if __name__ == "__main__":
    db = lambda y: mydb.create_engine('boe',f'd506{y}')

    with open('pop.html','r') as f:
        pophtml = f.read()


    soup = BeautifulSoup(pophtml,'html.parser')
    p = soup.find('table',{'class':'pagemain'})
    tr = p.findChild('tbody').findChildren('tr')[13:64]
    tage = pd.DataFrame([i.text.split('\n')[1:-1] for i in tr])
    tage = pd.concat([
        tage[[0,3]]\
            .rename(columns={0:'age',3:'n'}),
        tage[[4,7]]\
            .rename(columns={4:'age',7:'n'})
        ])
    age = tage['age']\
        .str.replace('น้อยกว่า 1 ปี','0 ปี')\
        .str.replace('มากกว่า 100 ปี','101 ปี')\
        .str.split(' ',expand=True)[0]
    tage['age'] = age.astype('int')
    tage = tage.set_index('age')['n']\
        .sort_index()\
        .str.replace(',','')\
        .astype('int')

    pop = pd.read_sql_query(popquery,db('63'))
    pop['provname'] = pop['provname'].str.encode('latin1').str.decode('tis620')
    pop['provcode'] = pop['provcode']


    df = None
    cyear = 63
    print('Fetching data')
    for y in tqdm(range(cyear-5,cyear+1)):
        if y == 63:
            query = pquery\
                .replace('{y}',str(y))\
                .replace('{WHERE}',
                        f'WHERE DATESICK < "{lastdate}"')
        else:
            query = pquery\
                .replace('{y}',str(y))\
                .replace('{WHERE}','')
        temp = pd.read_sql_query(query,db(y))
        df = pd.concat([df,temp])
    df = df.reset_index(drop=True)\
        .assign(DATESICK = lambda x: pd.to_datetime(x['DATESICK']))\
        .assign(WEEK = lambda x: x['DATESICK'] - pd.to_timedelta(x['DATESICK'].dt.dayofweek+1,'D'))\
        .rename(columns={'YEAR':'AGE'})\
        .assign(YEAR = lambda x: x['DATESICK'].dt.year)

    dateSer = pd.Series(
        np.arange('2014-01-01','2020-12-31',
        dtype='datetime64')
        )
    weekSer = (
        dateSer-pd.to_timedelta(dateSer.dt.dayofweek+1,'D')
        )
    weekdf = pd.DataFrame(weekSer[weekSer.dt.year >= 2014]\
                .rename('WEEK'))\
                .assign(YEAR = lambda x: x['WEEK'].dt.year)
    firstdate = weekdf.groupby(['YEAR']).min()\
                .rename(columns={'WEEK':'firstdate'})
    weekdf = weekdf.set_index('YEAR')\
                .merge(
                    firstdate,
                    left_index=True,
                    right_index=True)\
                .assign(DAYS = lambda x: x['WEEK']-x['firstdate'])\
                .reset_index()\
                .drop_duplicates()\
                .set_index(['DAYS','YEAR'])['WEEK']\
                .unstack()\
                .reset_index(drop=True)
    weekdf.columns.name = 'EYEAR'
    weekdf.index = weekdf.index + 1
    weekdf.index.name = 'WEEKNO'
    weekdf = weekdf.stack().rename('WEEK').reset_index()
    df['WEEK'] = pd.to_datetime(df['WEEK'])
    dfwithwekkno = df.merge(weekdf,left_on='WEEK',right_on='WEEK',how='left')

    ts = dfwithwekkno.groupby(['WEEK'])['CASES'].sum()
    yearweekcase = dfwithwekkno\
                    .query('PROVINCE == "10"')\
                    .groupby(
                        ['EYEAR','WEEKNO'],
                        as_index=False)['CASES']\
                    .sum()\
                    .astype('int')
    go_yearweekcase = [
        go.Scatter(
            x=yearweekcase.query('EYEAR == @year')['WEEKNO'],
            y=yearweekcase.query('EYEAR == @year')['CASES'],
            name=year
            ) \
                for year in range(2015,2021)]

    dfwithmedian = dfwithwekkno\
                    .groupby(['EYEAR','PROVINCE','WEEKNO'])['CASES']\
                    .sum()\
                    .astype('int')\
                    .unstack('EYEAR')\
                    .assign(median = lambda x: x.loc[:,2015:2019]\
                    .median(axis=1))\
                    .loc[:,[2020,2019,'median']]

    last_2week = int(dfwithmedian[2020]\
                    .unstack()\
                    .sum(axis=0)\
                    .replace(0,np.nan)\
                    .dropna()\
                    .index[-1])

    plist = dfwithmedian\
                    .loc[:,[2020,'median']]\
                    .unstack('PROVINCE')[:last_2week]\
                    .sum(axis=0).unstack('EYEAR')\
                    .assign(median_ratio_2020 = lambda x: x['median']/x[2020])\
                    .sort_values('median_ratio_2020')\
                    .query('median_ratio_2020 != inf')\
                    .index


    """
    By Week
    """
    all_th = dfwithmedian\
                .stack().unstack('PROVINCE').sum(axis=1).unstack()\
                .assign(PROVINCE = '999')\
                .reset_index()\
                .set_index(['PROVINCE','WEEKNO'])

    dfwithmedianwithth = pd.concat([all_th,dfwithmedian])
    pop = pop.sort_values('provcode')
    thpop = pd.DataFrame([{'provcode':'999','provname':'ทั่วประเทศ'}])
    popwithth = pd.concat([thpop,pop])
    provs = popwithth['provcode']
    data = [casebyprov(str(prov),dfwithmedianwithth) for prov in provs]
    plot_data = plotweek(data[0],visible=True)
    for datai in data[1:]:
        plot_data = plot_data+plotweek(datai,visible=False)
    buttons = []
    for i,prov in enumerate(provs):
        temp = dict(
            label = popwithth['provname'].iloc[i],
            method = 'update',
            args = [{'visible': [list(map(lambda x: [prov==x,prov==x,prov==x],provs))]},
                    {'title':popwithth['provname'].iloc[i]}])
        buttons.append(temp)
    for button in buttons:
        nested = button['args'][0]['visible']
        button['args'][0]['visible'] = [reduce(lambda x,y: x+y,x) for x in nested][0]

    updatemenus = list([
        dict(active=0,
            #type='buttons',
            buttons=buttons,
            #direction="left",
            pad={"r": 100, "t": 10},
            showactive=True,
            x=0.3,
            xanchor="left",
            y=1.4,
            yanchor="top")
            ])

    layout = dict(
        showlegend=True,
        updatemenus=updatemenus,
        height=250,
        margin={"r":0,"t":40,"l":0,"b":0}
        )
    fig = dict(data=plot_data, layout=layout)
    div = go_to_div(fig)

    nationmedian = dfwithmedian.stack().unstack('PROVINCE').sum(axis=1).unstack()
    nationmedian.index = nationmedian.index.astype('int')

    with open('plotly_th.html','w') as f:
        f.write(div)


    fig = make_subplots(rows = 16, cols = 5)
    for i,prov in enumerate(pop['provcode'].sort_values()):
        r = int(np.floor(i/5))+1
        c = (i%5)+1
        fig.add_trace(go.Scatter(x=dfwithmedian.loc[prov].index,
                                y=dfwithmedian.loc[prov][2019],
                                name=f'2019{prov}',marker_color='green'),
                row=r,col=c)
        fig.add_trace(go.Scatter(x=dfwithmedian.loc[prov].index,
                                y=dfwithmedian.loc[prov]['median'],
                                name=f'median{prov}',marker_color='red'),
            row=r,col=c)
        fig.add_trace(go.Scatter(x=dfwithmedian.loc[prov].index,
                                y=dfwithmedian.loc[prov][2020],
                                name=f'2020{prov}',marker_color='blue',marker_size=10),
                    row=r,col=c)

    fig.update_layout(height=1800,showlegend=False)
    div = plotly.offline.plot(fig,
                        include_plotlyjs=False,
                        output_type='div')
    script = f'<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>{div}'
    with open('plotly_allp.html','w') as f:
        f.write(div)
    caseprov = df[df['DATESICK'].dt.year == 2020].groupby('PROVINCE')['CASES'].sum()
    caseprovrate = pop\
                    .set_index('provcode')\
                    .merge(
                        caseprov,
                        left_index=True,
                        right_index=True)\
                    .assign(rate = lambda x: x['CASES']/x['pop']*100000)\
                    .loc[:,'rate']\
                    .reset_index()\
                    .rename(columns={'index':'provcode'})

    mongocursor = pymongo.MongoClient().maps.province.find({},{'_id':0})
    provincejson = mongocursor[0]
    for feature_prov in provincejson['features']:
        feature_prov['id'] = feature_prov['properties']['PROVINCE']

    fig = px.choropleth(caseprovrate,
                geojson=provincejson,
                locations='provcode',
                color='rate',
                color_continuous_scale='Reds',
                scope='asia'
                )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(width=1000,height=500,margin={"r":0,"t":0,"l":0,"b":0})
    div = plotly.offline.plot(fig,
                        include_plotlyjs=False,
                            config={'scrollZoom': False,'staticPlot':False},
                        output_type='div')
    script = f'<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>{div}'
    with open('plotly_map.html','w') as f:
        f.write(div)

    y2020 = df[df['DATESICK'].dt.year == 2020]
    y2020death = y2020.query('RESULT == 2')
    totalcase = (len(y2020))
    totaldeath = len(y2020death)
    y2020age = y2020.groupby('AGE')['CASES'].sum()
    y2020age = pd.concat([y2020age,tage],axis=1).dropna()
    g1 = (y2020age.index < 5) * 1 
    g2 = ((y2020age.index >= 5) & (y2020age.index < 15)) * 2
    g3 = ((y2020age.index >= 15) & (y2020age.index < 25)) * 3
    g4 = ((y2020age.index >= 25) & (y2020age.index < 35)) * 4
    g5 = ((y2020age.index >= 35) & (y2020age.index < 45)) * 5
    g6 = ((y2020age.index >= 45) & (y2020age.index < 55)) * 6
    g7 = ((y2020age.index >= 55) & (y2020age.index < 65)) * 7
    g8 = (y2020age.index >= 65) * 8
    agegr = g1+g2+g3+g4+g5+g6+g7+g8
    y2020age['agegr'] = agegr.astype('int')
    agegrtxt = [
        '0-4 ปี',
        '5-14 ปี',
        '15-24 ปี',
        '25-34 ปี',
        '35-44 ปี',
        '45-54 ปี',
        '55-64 ปี',
        '65+ ปี'
    ]

    agegrser = pd.Series(agegrtxt,index=np.arange(1,9))
    y2020age = y2020age\
                .assign(agegrtxt = lambda x: x['agegr'].replace(agegrser.to_dict()))
    byagegr = y2020age\
                .groupby(['agegr','agegrtxt'],as_index=False)\
                .sum()\
                .assign(rate = lambda x: np.round(x['CASES']/x['n']*100000,2),
                        rate2 = lambda x:\
                                        "กลุ่มอายุ " + x.index.astype('str') +\
                                        " จำนวน " + x['CASES'].astype('int').astype('str') + ' คน (' +\
                                x['rate'].astype('str') + ")")\
                .set_index('agegrtxt')
    agegrlist = byagegr.sort_values('rate',ascending=False).iloc[:3]['rate2']
    agegrhtml = ", ".join(agegrlist)

    fig = go.Figure(go.Bar(y=byagegr['rate'],x=byagegr.index))
    fig.update_layout(width=600,height=200,margin={"r":0,"t":0,"l":0,"b":0})
    div = plotly.offline.plot(fig,
                        include_plotlyjs=False,
                            config={'scrollZoom': False,'staticPlot':False},
                        output_type='div')
    script = f'<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>{div}'
    with open('plotly_agegr.html','w') as f:
        f.write(div)



    with open('framework.html','r') as f:
        fw = f.read()



    with open('plotly_map.html','r') as f:
        mapjs = f.read()
    with open('plotly_th.html','r') as f:
        mapth = f.read()
    with open('plotly_allp.html','r') as f:
        pall = f.read()
    with open('plotly_agegr.html','r') as f:
        agegrplot = f.read()
    #mapjs = mapjs.replace('height:800px; width:1500px','height:300px; width:300px')
    dstart = (pd.to_datetime(dlast) - pd.to_timedelta(6,'D'))
    dstart_th = thai_strftime(dstart,'%d %B %Y')
    dstop_th = thai_strftime(pd.to_datetime(dlast),'%d %B %Y')
    ndays = dstart - pd.to_datetime('2020-01-05')
    cweek = int((ndays.days/7)+1)
    html = fw.replace('{trendline}',mapth)\
            .replace('{map}',mapjs)\
            .replace('{agegrplot}',agegrplot)
    html = html.replace('{cweek}',str(cweek))


    prolist = caseprovrate\
                .merge(
                    pop,
                    left_on='provcode',
                    right_on='provcode',
                    how='left')\
                .assign(rate = lambda x: np.round(x['rate'],2))\
                .loc[:,['provname','rate']]\
                .sort_values('rate',ascending=False)\
                .assign(wording = lambda x: 'จังหวัด'+x['provname']+' ('+x['rate'].astype('str')+')')\
                .iloc[:10]\
                .loc[:,'wording']
    provincehtml = " ".join(prolist)
    regions = pd.read_sql_query('select region,provcode from province',db('63'))
    regions['region'].value_counts()


    popser = pop.sort_values('provcode')['pop']
    dfprov2020 = dfwithmedian[2020].unstack()[~dfwithmedian[2020].unstack().index.isin(['1','2'])]
    dfprov2020 = pd.DataFrame(100000*dfprov2020.values / popser.values.reshape(77,1),
                            columns=dfprov2020.columns,index=dfprov2020.index)
    zmax = dfprov2020.max().max()

    """
    North
    """
    pcodenorth = regions.query('region == "north"')['provcode']
    areamat = dfprov2020[dfprov2020.index.isin(pcodenorth)]
    areamat = areamat.merge(pop,left_index=True,right_on='provcode').set_index('provname').iloc[:,:-2]
    go_n = go_to_div(go.Figure(go.Heatmap(
                                z=areamat,
                                y=areamat.index,
                                x=areamat.columns,
                                colorscale='reds',
                                zauto=False,
                                zmax=zmax))\
                        .update_layout(
                                    height=300,
                                    margin={"r":0,"t":0,"l":0,"b":0})
                    )

    """
    Northeast
    """
    pcodenorth = regions.query('region == "northeast"')['provcode']
    areamat = dfprov2020[dfprov2020.index.isin(pcodenorth)]
    areamat = areamat.merge(pop,left_index=True,right_on='provcode').set_index('provname').iloc[:,:-2]
    go_ne = go_to_div(go.Figure(go.Heatmap(z=areamat,
                                        y=areamat.index,
                                        x=areamat.columns,
                                        colorscale='reds',zauto=False,zmax=zmax))\
                    .update_layout(height=300,margin={"r":0,"t":0,"l":0,"b":0}))

    """
    Central
    """
    pcodenorth = regions.query('region == "central"')['provcode']
    areamat = dfprov2020[dfprov2020.index.isin(pcodenorth)]
    areamat = areamat.merge(pop,left_index=True,right_on='provcode').set_index('provname').iloc[:,:-2]
    go_c = go_to_div(go.Figure(go.Heatmap(z=areamat,
                                        y=areamat.index,
                                        x=areamat.columns,
                                        colorscale='reds',zauto=False,zmax=zmax))\
                    .update_layout(height=300,margin={"r":0,"t":0,"l":0,"b":0}))

    """
    South
    """
    pcodenorth = regions.query('region == "south"')['provcode']
    areamat = dfprov2020[dfprov2020.index.isin(pcodenorth)]
    areamat = areamat.merge(pop,left_index=True,right_on='provcode').set_index('provname').iloc[:,:-2]
    go_s = go_to_div(go.Figure(go.Heatmap(z=areamat,
                                        y=areamat.index,
                                        x=areamat.columns,
                                        colorscale='reds',zauto=False,zmax=zmax))\
                    .update_layout(height=300,margin={"r":0,"t":0,"l":0,"b":0}))

    lastdate = df[df['DATESICK'].dt.year == 2020]['DATESICK'].max()
    thlastdate = thai_strftime(lastdate,'%d %B %Y')
    thstartweek = thai_strftime(df[df['DATESICK'].dt.year == 2020]['WEEK'].max(),'%d %B %Y')


    totalrate = np.round(totalcase/pop.sum()['pop']*100000,2)
    last4 = nationmedian[2020][last_2week-4:last_2week].sum()
    last4med = nationmedian['median'][last_2week-4:last_2week].sum()
    last8 = nationmedian[2020][last_2week-8:last_2week-4].sum()
    ptrend = last4/last8
    pmed = last4/last4med
    if ptrend < 0.8:
        trend = 'ลดลง'
    elif ptrend > 1.2:
        trend = 'เพิ่มขึ้น'
    else:
        trend = 'คงที่'
    if pmed < 0.8:
        withmedian = 'ต่ำกว่า'
    elif pmed > 1.2:
        withmedian = 'มากกว่า'
    else:
        withmedian = 'ไม่แตกต่าง'

    html = html\
        .replace('{dstart}',dstart_th)\
        .replace('{dstop}',dstop_th)\
        .replace('{totalcase}',str(totalcase))\
        .replace('{agegrhtml}',str(agegrhtml))\
        .replace('{totaldeath}',str(totaldeath))\
        .replace('{totalrate}',str(totalrate))\
        .replace('{trend}',trend)\
        .replace('{withmedian}',withmedian)\
        .replace('{provinces}',provincehtml)\
        .replace('{go_n}',go_n)\
        .replace('{go_ne}',go_ne)\
        .replace('{go_c}',go_c)\
        .replace('{go_s}',go_s)

    with open(f'report_{ds}_{dlast}.html','w') as f:
        f.write(html)