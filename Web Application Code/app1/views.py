from django.http import JsonResponse
from django.shortcuts import render
import pickle as pk
import pandas as pd

from sklearn.preprocessing import StandardScaler



def home(request):

    return render(request,"app1/landing_page.html",context={})


def index(request):
    if request.method == 'POST':
        brand_name = request.POST['psname']
        df = pd.read_csv("media/Project_data6.csv")
        gk = df.groupby('Brand')['Model'].agg(['unique']).apply(list).to_dict()
        context = {'data': dict(enumerate(gk['unique'][brand_name].flatten(), 1))}

        print(context)
        print(request.POST['psname'])
        return JsonResponse({"data": context}, status=200)

    return render(request, "app1/index.html", context={})


def motor(request):
    if request.method == 'POST':
        brand = request.POST['brand']
        model = request.POST['model']
        year = int(request.POST['year'])
        color = request.POST['color']
        fuel = request.POST['fuel']
        gear = request.POST['gear']
        engine = float(request.POST['engine'])
        mileage = float(request.POST['mileage'])
        region = request.POST['region']
        print(brand,' ',model,' ',year,' ',color,' ',fuel,' ',gear,' ',engine,' ',mileage,' ',region)
        dataset = pd.read_csv('media/Project_data6.csv')

        pr1 = pd.DataFrame([[brand, model, year, color, fuel, gear, engine, mileage, region]]
                           , columns=["Brand", "Model", "Year", "Color", "fuelType", "gearType", "engineSize",
                                      "Mileage", "Region"])

        df = [dataset, pr1]
        dataset = pd.concat(df)

        dataset = dataset[dataset["Mileage"] < 900000]
        sc1 = StandardScaler()
        dataset['Mileage'] = sc1.fit_transform(dataset[['Mileage']])
        dataset['Age'] = 2023 - dataset['Year']

        x = dataset[['Age', 'engineSize', 'Mileage', 'Brand', 'Color', 'gearType', 'Region',
                     'Model', 'fuelType']]

        x = pd.get_dummies(x, columns=['Brand', 'Color', 'gearType', 'Region',
                                       'Model', 'fuelType'])

        x_pred = x.iloc[[-1]].values

        regressor = pk.load(open('media/regressor.pkl', 'rb'))

        a = regressor.predict(x_pred)
        numbers = "{:,}".format(round(a[0], 2))
        context = {'result': numbers}

        return JsonResponse(context, status=200)

    return render(request, "app1/index.html", context={})


