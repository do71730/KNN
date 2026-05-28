#!pip install kagglehub
#import kagglehub
import pandas as pd
# Download latest version

def Euclidian_dist(p1,p2) -> float:
    dist = (abs(p1[0] - p2[0])**2 + abs(p1[1] - p2[1])**2) ** 0.5
    return dist

def sorting(pt,points):
    pt_dict = dict.fromkeys(points[pt],0)
    pt_dict = {points[pt][i]:Euclidian_dist(pt,points[pt][i]) for i in range(len(points[pt]))}
    pt_dict = dict(sorted(pt_dict.items(), key=lambda item: item[1], reverse=False))
    return list(pt_dict.keys())

def find_neighbors(k,points):
    points_dist = list(points.keys())
    for i in range(len(points_dist)):
        
        for j in range(i+1,len(points_dist)):

            if len(points[points_dist[j]]) > 0:
                points[points_dist[j]]= sorting(points_dist[j],points)
            if len(points[points_dist[i]]) > 0:
                points[points_dist[i]]= sorting(points_dist[i],points)

            if points_dist[j] not in points[points_dist[i]]:
                p = Euclidian_dist(points_dist[i],points_dist[j])

                if len(points[points_dist[i]]) < k:
                    points[points_dist[i]].append(points_dist[j])
                    points[points_dist[i]]= sorting(points_dist[i],points)

                    if points_dist[i] not in points[points_dist[j]]:
                        points[points_dist[j]].append(points_dist[i])
                        if len(points[points_dist[j]]) > k:
                            points[points_dist[j]] = sorting(points_dist[j],points)[:-1]
                        else:
                            points[points_dist[j]] = sorting(points_dist[j],points)
                        
                else:
                    if points_dist[j] not in points[points_dist[i]]:
                        points[points_dist[i]].append(points_dist[j])
                        points[points_dist[i]]= sorting(points_dist[i],points)[:-1]
                    if points_dist[i] not in points[points_dist[j]]:
                        points[points_dist[j]].append(points_dist[i])
                        if len(points[points_dist[j]]) > k:
                            points[points_dist[j]] = sorting(points_dist[j],points)[:-1]
                        else:
                            points[points_dist[j]] = sorting(points_dist[j],points)
    return points



def voting(point):
    points_dist = list(point.keys())
    y_pred = {}
    y = ""
    for x in points_dist:
        labels = {}
        y = point[x][0][-1]

        for i in range(len(point[x])):
            lbl = point[x][i][-1]

            if lbl in labels.keys():
                labels[lbl] += 1
            else:
                labels[lbl] = 1
            if labels[lbl] > labels[y]:
                y = lbl
            elif labels[lbl] == labels[y]:
                y = point[x][0][-1]

        y_pred[x] = y
        #print(y_pred)
    return y_pred

def model_elv(y_pred):
    TP = 0 #is a got a
    FN = 0 #predit birch but class is pear, false negative for pear
    FP = 0 #predit birch but class is pear, false positive for birch
    print(y_pred)
    
    classes = {k[-1]:{'TP':0,'FN':0,'FP':0} for k in y_pred.keys()}

    for x in list(y_pred.keys()):
        y = x[-1]
        y_hat = y_pred[x]
        if y == y_hat:
            classes[y]['TP'] += 1
            TP +=1
        else:
            classes[y_hat]['FP'] += 1
            classes[y]['FN'] += 1
    precision = 0
    recall = 0
    for c in classes.keys():
        tp = classes[c]['TP']
        fp = classes[c]['FP']
        fn = classes[c]['FN']
        if tp+fp != 0:
            precision += tp/(tp+fp)
        if tp+fn != 0:
            recall += tp/(tp+fn)
    precision = precision/len(list(classes.keys()))
    recall = recall/len(list(classes.keys()))

    print("Accuracy: ", TP/len(list(y_pred.keys())))
    print("Precision: ", precision)
    print("Recall: ",  recall)

def knn(k,points):
    #Euclidian_dist(list(points.keys())[0],list(points.keys())[100])
    p = find_neighbors(k,points)
    r = voting(p)
    print(model_elv(r))

def main():
    #path = kagglehub.dataset_download("khushikyad001/urban-tree-health-monitoring-dataset")
    #print("Path to dataset files:", path)
    df = pd.read_csv("c:/Users/princ/.cache/kagglehub/datasets/khushikyad001/urban-tree-health-monitoring-dataset/versions/1/urban_tree_health_monitoring.csv")
    print(df.head())
    '''points = {
    (37.649912, -122.163421, 'Oak'): [],
    (37.880514, -122.101442, 'Oak'): [],
    (37.792566, -122.374982, 'Pine'): [],
    (37.739238, -122.187811, 'Birch'): [],
    (37.562689, -122.214389, 'Birch'): [],
    (37.562144, -122.083812, 'Pine'): [],
    (37.523471, -122.047188, 'Elm'): [],
    (37.846238, -122.493702, 'Redwood'): [],
    (37.740672, -122.162744, 'Oak'): [],
    (37.783011, -122.473841, 'Oak'): []
    }'''
    points = dict.fromkeys(list(zip(df['Latitude'],df['Longitude'],df['Species'])),[])
    k = 20
    knn(k,points)


if '__init__' == main():
    main()