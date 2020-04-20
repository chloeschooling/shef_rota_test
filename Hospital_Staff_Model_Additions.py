import matplotlib.pyplot as plt
plt.style.use('bmh')
import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import figure
import random
from random import seed
from random import randint
import math
import statistics
mpl.rcParams['legend.numpoints'] = 1

#-----Same model with 3 additions------
#-1 Introducing risk of self-isolation
#-2 Time dependent infection/isolation rates
#-3 New assigment policy when rearranging to critical area

loop = 100 #Number of times to repeat random sampling of Monte Carlo Model. 100 chosen to be appropriate.
N = 30 # Paramter 1: Total number of staff on rota
n = 7 # Paramter 2: Length of each shift (days)
T = 196 # Paramter 3: Time duration for which the model is run for (days)
nG = 3 #  Paramter 4: Number of different areas
sigmoid_data = [] #Curve which is an approximation of how infection rates may change over time

for x in range(1, 30):
    sig = (math.exp(0.3*(x+15))-0.95)
    sigmoid_data.append(0.00000018*sig)
for x in range(1, 14):
    sig = (math.exp(0.2*x)-math.exp(-0.2*x))/(math.exp(0.2*x)+math.exp(-0.2*x))
    sigmoid_data.append(0.507*sig)
for x in range(1, 110):
    sig = (math.exp(0.1*(37-x))-math.exp(-0.1*(37-x)))/(math.exp(0.1*(37-x))+math.exp(-0.1*(37-x)))
    sigmoid_data.append(0.229*sig + 0.274)
for x in range(1, 150):
    sig = (1/(1+ math.exp(0.15*(-40 +x))))
    sigmoid_data.append(0.0457*sig)

def multiply_list(list, m): #Function to multiply all elements in a list by a factor m
    out = []
    for x in range(len(list)):
        out.append(m*list[x])
    return out

perc = []
each =[multiply_list(sigmoid_data, 6.2), multiply_list(sigmoid_data, 0.11), multiply_list(sigmoid_data, 0.01)]
for i in range(nG):
    perc.append(each[i]) #  Paramter 5: Infection risk for each area of work (%) [Area 1, Area2, Area3]
off_perc = multiply_list(sigmoid_data, 0.01)#  Paramter 6: Background infection risk (%)

off=[]
DAYoff = [0, 2, 2] #  Parameter 7: Number of days off per shift in each area [Area 1, Area2, Area3]

for i in range(nG):
    off.append(DAYoff[i])

perc_iso = multiply_list(sigmoid_data, 2) #  Parameter 8: Self-isolation risk (%)
perc_iso_inf = 0.11 #  Parameter 9: Risk of infection while self isolating (%)
n_key = 1 #  Paramter 10: Optionally chose a critical area. 1 = Area1, 2 = Area2, 3 = Area3
lim = 6 #  Paramter 11: Critical number of staff needed in the critical area

person = [[[] for x in range(N)] for y in range(loop)] # multi-dimensional list of all staff members, with their avaialibilty over the time period
person2 = [0 for x in range(loop)] # Save staff avaialibilty for a second time if we redistribute to critical area
group = [[] for x in range(N)] #Corresponding list to show which area each staff member is in on each day
infection_day = [0 for x in range(N)] #The day each staff member is infected
incub_store = [0 for x in range(N)] #The incubation period for each staff member
groups = ['Area ' + str(x+1) for x in range(nG)] #List of areas
time_off = ['off ' + str(x+1) for x in range(nG)]
incubation = [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 11, 12] #Possible incubation periods list, fitted to reported data
days_on = [[[] for x in range(N)] for y in range(loop)] #List to store which days work and which days off work
longest_run = [[0 for x in range(N)] for y in range(loop)] #List to store maximum number of days worked
Helpers = [[0 for x in range(N)] for y in range(loop)] #List to store how many times moved to help critical area
count_rest = [[0 for x in range(N)] for y in range(loop)] #List to store how many days off work
count_intense = [[0 for x in range(N)] for y in range(loop)] #List to count how many days of intense work (in critical area)
infection_day = [0 for x in range(N)] #The day each staff member is infected

def when_infect_time(p, p_off, p_iso, p_iso_inf): #Function to  determine  what day someone is infected accounting for time dependent infection and isolation risk
    infect = -100
    iso = -100
    start = 0
    for a in range(int(T/(n*nG))):
        for b in range(nG):
            for i in range(start*n, (start+1)*n - off[b]):
                if(iso<i+14):
                    if (random.randint(0,100000) < 1000*p[b][i]):
                        if((infect<0)):
                           infect = i
                else:
                    if(random.randint(0,100000) < 1000*p_iso_inf):
                        if((infect<0)):
                           infect = i
                if(infect!=i):
                    if(random.randint(0,100000) < 1000*p_iso[i]):
                        if((iso<0) & (infect<0)):
                            iso = i
            for i in range((start+1)*n - off[b], (start+1)*n):
                if(iso<i+14):
                    if (random.randint(0,100000) < 1000*p_off[i]):
                        if((infect<0) &(iso<i+14)):
                               infect = i
                else:
                    if(random.randint(0,100000) < 1000*p_iso_inf):
                        if((infect<0)):
                           infect = i
                if(infect!=i):
                    if(random.randint(0,100000) < 1000*p_iso[i]):
                        if((iso<0) & (infect<0)):
                            iso = i
            start+=1
            if((infect>0)&(iso>0)):
                break
    r = random.randint(0, len(incubation) - 1)
    incub = incubation[r] #Determine incubation length
    return([infect, iso, incub]) #Return day of infection, day of self isolation, length of incubation

def column(matrix, i): #Function to return column i of a 2d matrix
    return [row[i] for row in matrix]

def how_tired(DaysOn, m): #Function to calculate maximum number of days worked
    add = 0
    for x in range(1, m):
        if(DaysOn[m-x]==1):
            add+=1
        else:
            break
    return(add)

def Not_In(i, list): #Function to determine if element i is in list
    correct = 1
    for x in range(len(list)):
        if(list[x]==i):
            correct=0
    return(correct)

def largest_row_of_ones(l): #Function to determine longest row of 1s in list l
    c = 0
    max_count = 0
    for j in l:
        c = c + 1 if j==1 else 0  # in other languages there is a ternary ?: op
        max_count = max( max_count, c)
    return max_count

#Lists to store outputs from each run
TOTAL = []
TOTAL_COL = [[] for x in range(nG)]
TOTAL2 = []
TOTAL_COL2 = [[] for x in range(nG)]

for Q in range(loop):  #Loop over model many times to generate  monte  carlo simulation
    for a in range(N): #Loop through all staff
        for X in range(nG):
            if((a>=X*N/nG) & (a<(X+1)*N/nG)): #First section of staff start shift in Area 1, next section start in Area 2 etc
                for b in range(int(T/n*nG)):
                    for c in range(X, nG):
                        for j in range(n-off[c]):
                            group[a].append(groups[c])
                        for j in range(off[c]):
                            group[a].append(time_off[c])
                    for c in range(0, X):
                        for j in range(n-off[c]):
                            group[a].append(groups[c])
                        for j in range(off[c]):
                            group[a].append(time_off[c])
                p = []
                for i in range(X, nG):
                    p.append(perc[i])
                for i in range(0, X):
                    p.append(perc[i])
                [infect, iso, incub] = when_infect_time(p, off_perc, perc_iso, perc_iso_inf)

                if(infect>=0):
                    if(iso<0):
                        infection_day[a] = infect
                        incub_store[a] = incub
                        for k in range(infect+incub): #available to work until day of infection plus incubation period
                            if(k<T):
                                person[Q][a].append(1)
                        for k in range(14): #unavailable to work for 14 days once symptomatic
                            if(infect+incub+k<T):
                                person[Q][a].append(0)
                        if(random.randint(0,100) < 80): #80% chance of returning after 2 weeks
                            if(infect+incub+14<T):
                                for k in range(T-(14+infect+incub)):
                                    person[Q][a].append(1)
                        else:
                            person[Q][a].append(0)
                            for b in range(27):
                                if(random.randint(0,10000) < 655): #Same chance of returning again each day for next 4 weeks. Probability leaves 3% never returning
                                    if(infect+incub+15+b<T):
                                        for k in range(T-(15+b+infect+incub)):
                                            person[Q][a].append(1)
                                    break
                                else:
                                    if(infect+incub+15+b<T):
                                        person[Q][a].append(0)
                                if(b==26):
                                    if(infect+incub+42<T):
                                        for k in range(T - 42 - (infect+incub)):
                                            person[Q][a].append(0)
                    if(iso>=0):
                        infection_day[a] = infect
                        incub_store[a] = incub
                        for k in range(0, iso):
                            person[Q][a].append(1)
                        if(iso+14<infect):
                            for k in range(iso, iso+14):
                                person[Q][a].append(0)
                            for k in range(iso + 14, infect):
                                person[Q][a].append(1)
                        else:
                            for k in range(iso, infect):
                                person[Q][a].append(0)
                        for k in range(infect, infect+incub): #available to work until day of infection plus incubation period
                            if(k<T):
                                person[Q][a].append(1)
                        for k in range(14): #unavailable to work for 14 days once symptomatic
                            if(infect+incub+k<T):
                                person[Q][a].append(0)
                        if(random.randint(0,100) < 80): #80% chance of returning after 2 weeks
                            if(infect+incub+14<T):
                                for k in range(T-(14+infect+incub)):
                                    person[Q][a].append(1)
                        else:
                            person[Q][a].append(0)
                            for b in range(27):
                                if(random.randint(0,10000) < 655): #Same chance of returning again each day for next 4 weeks. Probability leaves 3% never returning
                                    if(infect+incub+15+b<T):
                                        for k in range(T-(15+b+infect+incub)):
                                            person[Q][a].append(1)
                                    break
                                else:
                                    if(infect+incub+15+b<T):
                                        person[Q][a].append(0)
                                if(b==26):
                                    if(infect+incub+42<T):
                                        for k in range(T - 42 - (infect+incub)):
                                            person[Q][a].append(0)
                if(infect<0):
                    if(iso<0):
                        infection_day[a] = 'never'
                        for k in range(T):
                            person[Q][a].append(1)
                    if(iso>=0):
                        for k in range(0, iso):
                            person[Q][a].append(1)
                        for k in range(iso, iso+14):
                            if(k<T):
                                person[Q][a].append(0)
                        if(iso+14<T):
                            for k in range(iso + 14, T):
                                person[Q][a].append(1)

    for a in range(N):
        for y in range(T):
            if(str(group[a][y])[0]=="A"):
                days_on[Q][a].append(1)
            if(str(group[a][y])[0]=="o"):
                count_rest[Q][a]+=1
                days_on[Q][a].append(0)
            if(group[a][y]==groups[n_key-1]):
                count_intense[Q][a]+=1

    day = []
    total = []
    total_col = [[] for x in range(nG)]
    total_off = [[] for x in range(nG)]
    total_col_plot = [[] for x in range(nG)]
    id_col = [[] for x in range(nG)]
    id_off = [[] for x in range(nG)]
    amount_col = [[] for x in range(nG)]
    amount_off = [[] for x in range(nG)]

    col_g = [0 for x in range(T)]

    for j in range(T):
        day.append(j+1)
        day.append(j+2)
        col = column(person[Q], j) #List of available people on day j

        sum_col=0
        for x in range(len(col)):
            sum_col+=col[x]
        col_g[j] = column(group, j) #List of area distribution on day j
        total.append(sum_col) #Total number of staff available on day j
        total.append(sum_col)

        add_col = [0 for x in range(nG)]
        add_off = [0 for x in range(nG)]
        who_col = [[] for x in range(nG)]
        who_off = [[] for x in range(nG)]
        am_col = [[] for x in range(nG)]
        am_off = [[] for x in range(nG)]


        for k in range(len(col)):
            for a in range(nG):
                if(col_g[j][k] == groups[a]):
                     add_col[a]+=col[k]
                     who_col[a].append(k)
                     am_col[a].append(col[k])
                if(col_g[j][k] == time_off[a]):
                     add_off[a]+=col[k]
                     who_off[a].append(k)
                     am_off[a].append(col[k])

        for a in range(nG):
            total_col[a].append(add_col[a]) #Available staff working in specific area
            for b in range(2):
                total_col_plot[a].append(add_col[a] + add_off[a])
            total_off[a].append(add_off[a]) #Available staff off work in specific area
            id_col[a].append(who_col[a]) #List of which staff in specific area
            id_off[a].append(who_off[a]) #List of which staff having days off in specific area
            amount_col[a].append(am_col[a])
            amount_off[a].append(am_off[a])


    TOTAL.append(total) #Store total staff available over time period
    for a in range(nG):
        TOTAL_COL[a].append(total_col_plot[a]) #Store staff available in each area over time period

    if((n_key>=1) & (n_key<=nG)): #If critical area has been chosen
        for y in range(T):
             if(y>0):
                 if((total_col[n_key-1][y]<lim) & (total_off[n_key-1][y]==0)):
                     sum = 0
                     num = lim-total_col[n_key-1][y]
                     other = [-100 for x in range(2*nG)]
                     other_id = [-100 for x in range(2*nG)]
                     other_am = [-100 for x in range(2*nG)]

                     for l in range(nG):
                         if (l!=(n_key-1)):
                                 other[l] = (total_off[l][y])
                                 other_id[l] = (id_off[l][y])
                                 other_am[l] = (amount_off[l][y])

                     for l in range(nG):
                         if (l!=(n_key-1)):
                             other[l+nG] = (total_col[l][y])
                             other_id[l+nG] = (id_col[l][y])
                             other_am[l+nG] = (amount_col[l][y])
                     save_by = []
                     saver = []
                     for inc in range(1, 20): #Loop through number of times each individual has been redistributed
                       for u in range(8): #Loop through number of days worked in a row up to today
                         if(num>sum):
                             for x in range(len(other_am)):
                                 if(other[x]>0):
                                     for z in range(len(other_am[x])):
                                         if((Helpers[Q][other_id[x][z]]<inc) & (Not_In(other_id[x][z], saver)==1) & (how_tired(days_on[Q][other_id[x][z]], y) <=u)):
                                             person[Q][other_id[x][z]][y]-=other_am[x][z] #Person is moved  to critical area
                                             Helpers[Q][other_id[x][z]]+=(1)
                                             days_on[Q][other_id[x][z]][y]=1 #If they were on a day off they have another day on
                                             if(other_am[x][z]>0):
                                                 count_intense[Q][other_id[x][z]]+=1
                                             if(x<4):
                                                 count_rest[Q][other_id[x][z]]-=1
                                             if(len(id_col[n_key-1][y])==0):
                                                 id_col[n_key-1][y].append(0)
                                             person[Q][id_col[n_key-1][y][0]][y]+=other_am[x][z]
                                             sum+=other_am[x][z] #Sum how many people we add to critical area
                                             save_by.append(other_am[x][z])
                                             saver.append(other_id[x][z])
                                             if sum>= num: #Once we've reached our criteria we break
                                                 break
                                     if sum>= num:
                                         break
                             if sum>= num:
                                     break
                       if sum>= num:
                                break

                     for x in range(len(saver)):
                        if(infection_day[saver[x]]!='never'):
                            if(y < infection_day[saver[x]]):
                                if (random.randint(0,100000) < 1000*perc[n_key-1][y]):
                                    incub = incub_store[saver[x]]
                                    for k in range(1, incub):
                                        if((k+y)<T):
                                                person[Q][saver[x]][k+y] = 1
                                    for k in range(incub, incub+14):
                                        if((k+y)<T):
                                                person[Q][saver[x]][k+y]=0
                                        if(random.randint(0,100) < 80): #80% chance of returning after 2 weeks
                                                for k in range(incub+14, T-y):
                                                    if((k+y)<T):
                                                        person[Q][saver[x]][k+y] = 1
                                        else:
                                            if((incub+14+y)<T):
                                                person[Q][saver[x]][incub+14+y] = 0
                                            for b in range(27):
                                                if(random.randint(0,10000) < 655): #Same chance of returning again each day for next 4 weeks. Probability leaves 3% never returning
                                                        for k in range(incub+15+b, T-y):
                                                            if((k+y)<T):
                                                                person[Q][saver[x]][k+y] = 1
                                                        break
                                                else:
                                                    if(incub+15+b+y<T):
                                                        person[Q][saver[x]][incub+15+b+y] = 0
                                                if(b==26):
                                                    if(incub+42+y<T):
                                                        for k in range(incub+42, T - y):
                                                            if((k+y)<T):
                                                                person[Q][saver[x]][k+y] = 0

                        else:
                            if (random.randint(0,100000) < 1000*perc[n_key-1][y]):
                                incub = incub_store[saver[x]]

                                for k in range(1, incub):
                                    if((k+y)<T):
                                            person[Q][saver[x]][k+y] = 1
                                for k in range(incub, incub+14):
                                    if((k+y)<T):
                                            person[Q][saver[x]][k+y]=0
                                    if(random.randint(0,100) < 80): #80% chance of returning after 2 weeks
                                            for k in range(incub+14, T-y):
                                                if((k+y)<T):
                                                    person[Q][saver[x]][k+y] = 1
                                    else:
                                        if((incub+14+y)<T):
                                            person[Q][saver[x]][incub+14+y] = 0
                                        for b in range(27):
                                            if(random.randint(0,10000) < 655): #Same chance of returning again each day for next 4 weeks. Probability leaves 3% never returning
                                                    for k in range(incub+15+b, T-y):
                                                        if((k+y)<T):
                                                            person[Q][saver[x]][k+y] = 1
                                                    break
                                            else:
                                                if(incub+15+b+y<T):
                                                    person[Q][saver[x]][incub+15+b+y] = 0
                                            if(b==26):
                                                if(incub+42+y<T):
                                                    for k in range(incub+42, T - y):
                                                        if((k+y)<T):
                                                            person[Q][saver[x]][k+y] = 0


                     total_col = [[] for x in range(nG)]
                     total_off = [[] for x in range(nG)]
                     id_col = [[] for x in range(nG)]
                     id_off = [[] for x in range(nG)]
                     amount_col = [[] for x in range(nG)]
                     amount_off = [[] for x in range(nG)]
                     for j in range(T):
                            col = column(person[Q], j) #List of available people on day j
                            add_col = [0 for x in range(nG)]
                            add_off = [0 for x in range(nG)]
                            who_col = [[] for x in range(nG)]
                            who_off = [[] for x in range(nG)]
                            am_col = [[] for x in range(nG)]
                            am_off = [[] for x in range(nG)]

                            for k in range(len(col)):
                                for a in range(nG):
                                    if(col_g[j][k] == groups[a]):
                                         add_col[a]+=col[k]
                                         who_col[a].append(k)
                                         am_col[a].append(col[k])
                                    if(col_g[j][k] == time_off[a]):
                                         add_off[a]+=col[k]
                                         who_off[a].append(k)
                                         am_off[a].append(col[k])

                            for a in range(nG):
                                total_col[a].append(add_col[a]) #Available staff working in specific area
                                total_off[a].append(add_off[a]) #Available staff off work in specific area
                                id_col[a].append(who_col[a]) #List of which staff in specific area
                                id_off[a].append(who_off[a]) #List of which staff having days off in specific area
                                amount_col[a].append(am_col[a])#List of how many staff available on a day working in specific area
                                amount_off[a].append(am_off[a])#List of how many staff available on a day off in specific area

        person2[Q] = person[Q]

        for x in range(N):
            for y in range(T):
                person2[Q][x][y] = person[Q][x][y]

        total2 = []
        total_col_plot2 = [[] for x in range(nG)]
        for j in range(T):
            col = column(person2[Q], j) #List of available people on day j
            sum_col=0
            for x in range(len(col)):
                sum_col+=col[x]
            col_g[j] = column(group, j) #List of area distribution on day j
            total2.append(sum_col) #Total number of staff available on day j
            total2.append(sum_col)
            add_col = [0 for x in range(nG)]
            add_off = [0 for x in range(nG)]
            for k in range(len(col)):
                for a in range(nG):
                    if(col_g[j][k] == groups[a]):
                         add_col[a]+=col[k]

                    if(col_g[j][k] == time_off[a]):
                         add_off[a]+=col[k]
            for a in range(nG):
                for b in range(2):
                    total_col_plot2[a].append(add_col[a] + add_off[a])

        TOTAL2.append(total2) #Store total staff available over time period
        for a in range(nG):
            TOTAL_COL2[a].append(total_col_plot2[a]) #Store staff available in each area over time period after rearrangement

        for b in range(N):
            longest_run[Q][b] = (largest_row_of_ones(days_on[Q][b])) #Store maximum number of days worked for each person


#------Calculate mean and confidence intervals (2.5th and 97.5th percentile) from the 100 runs---------
mean_tot = []
upper_tot = []
lower_tot = []
opp_mean = []
opp_upper = []
opp_lower = []
mean_grp = [[] for x in range(nG)]
upper_grp = [[] for x in range(nG)]
lower_grp = [[] for x in range(nG)]

for y in range(2*T):
    list = column(TOTAL, y)
    mean = statistics.mean(list)
    mean_tot.append(mean)
    opp_mean.append(N - mean)
    stdev = np.std(list)
    upper_tot.append(np.percentile(list, 97.5))
    opp_upper.append(N - np.percentile(list, 2.5))
    lower_tot.append(np.percentile(list, 2.5))
    opp_lower.append(N - np.percentile(list, 97.5))

    for a in range(nG):
        list = column(TOTAL_COL[a], y)
        mean = statistics.mean(list)
        mean_grp[a].append(mean)
        stdev = np.std(list)
        upper_grp[a].append(np.percentile(list, 97.5))
        lower_grp[a].append(np.percentile(list, 2.5))

if((n_key>=1) & (n_key<=nG)):
    mean_tot2 = []
    upper_tot2 = []
    lower_tot2 = []
    opp_mean2 = []
    opp_upper2 = []
    opp_lower2 = []
    mean_grp2 = [[] for x in range(nG)]
    upper_grp2 = [[] for x in range(nG)]
    lower_grp2 = [[] for x in range(nG)]

    for y in range(2*T):
        list = column(TOTAL2, y)
        mean = statistics.mean(list)
        mean_tot2.append(mean)
        opp_mean2.append(N - mean)
        stdev = np.std(list)
        upper_tot2.append(np.percentile(list, 97.5))
        opp_upper2.append(N - np.percentile(list, 2.5))
        lower_tot2.append(np.percentile(list, 2.5))
        opp_lower2.append(N - np.percentile(list, 97.5))

        for a in range(nG):
            list = column(TOTAL_COL2[a], y)
            mean = statistics.mean(list)
            mean_grp2[a].append(mean)
            stdev = np.std(list)
            upper_grp2[a].append(np.percentile(list, 97.5))
            lower_grp2[a].append(np.percentile(list, 2.5))

#------Plot results---------

figure(figsize=(10, 5))
plt.fill_between(day, lower_tot, upper_tot, color = 'g', alpha =0.5, label = "95% confidence interval \navailable")
plt.plot(day, mean_tot, color = 'g', label = "Mean available")
plt.fill_between(day, opp_upper, opp_lower, color = 'b', alpha =0.5, label = "95% confidence interval \nunavailable")
plt.plot(day, opp_mean, color = 'b', label = "Mean unavailable")
legend = plt.legend(loc = 'best', fontsize = 'medium')
plt.xlabel('Days', fontsize =12)
plt.ylabel('Number of Staff', fontsize =12)
plt.title('Total number of staff')
plt.show()

if((n_key>=1) & (n_key<=nG)):
    figure(figsize=(10, 5))
    plt.fill_between(day, lower_tot2, upper_tot2, color = 'g', alpha =0.5, label = "95% confidence interval \navailable")
    plt.plot(day, mean_tot2, color = 'g', label = "Mean available")
    plt.fill_between(day, opp_upper2, opp_lower2, color = 'b', alpha =0.5, label = "95% confidence interval \nunavailable")
    plt.plot(day, opp_mean, color = 'b', label = "Mean unavailable")
    legend = plt.legend(loc = 'best', fontsize = 'medium')
    plt.xlabel('Days', fontsize =12)
    plt.ylabel('Number of Staff', fontsize =12)
    plt.title('Total number of staff (after rearrangement)')
    plt.show()


col = ['r',  'orange', 'purple', 'olive', 'navy', 'violet', 'turquoise', 'maroon']
for a in range(nG):
    figure(figsize=(10, 5))
    plt.fill_between(day, lower_grp[a], upper_grp[a], color = col[a], alpha =0.5, label = "95% confidence interval")
    plt.plot(day, mean_grp[a], color = col[a], label = "Mean value")
    legend = plt.legend(loc = 'best', fontsize = 'medium')
    plt.xlabel('Days', fontsize =12)
    plt.ylabel('Number of Staff available', fontsize =12)
    plt.ylim(0, max(upper_grp[a])+1)
    plt.title(groups[a])
    plt.show()

    if((n_key>=1) & (n_key<=nG)):
        figure(figsize=(10, 5))
        plt.fill_between(day, lower_grp2[a], upper_grp2[a], color = col[a], alpha =0.5, label = "95% confidence interval")
        plt.plot(day, mean_grp2[a], color = col[a], label = "Mean value")
        legend = plt.legend(loc = 'best', fontsize = 'medium')
        plt.xlabel('Days', fontsize =12)
        if(a==(n_key-1)):
            plt.axhline(y=lim, linewidth =1 , linestyle = "--", color="k", label = "Critical number in " + str(groups[a]))
        plt.ylabel('Number of Staff available', fontsize =12)
        plt.title(groups[a] + ' (after rearrangement)')
        plt.ylim(0, max(upper_grp2[a])+1)
        plt.show()

#------Calculate mean statitics on intensity of work and plot as histograms---------
mean_count_intense = []
mean_count_rest = []
mean_count_longrun = []
mean_count_manyHelps = []

for x in range(N):
    list = column(count_intense, x)
    mean = statistics.mean(list)
    mean_count_intense.append(100*mean/T)
    list = column(count_rest, x)
    mean = statistics.mean(list)
    mean_count_rest.append(100*mean/T)
    list = column(longest_run, x)
    mean = statistics.mean(list)
    mean_count_longrun.append(mean)
    list = column(Helpers, x)
    mean = statistics.mean(list)
    mean_count_manyHelps.append(mean)

x = mean_count_intense
plt.hist(x, range = (int(min(x)-1), int(max(x)+1)), bins=(10*int(int(max(x)+1) - int(min(x)-1))), color='0.7')
plt.ylabel('Frequency')
plt.xlabel('Percentage of days high risk work')
plt.show()

x = mean_count_rest
plt.hist(x, range = (int(min(x)-1), int(max(x)+1)), bins=(10*int(int(max(x)+1) - int(min(x)-1))), color='0.7')
plt.ylabel('Frequency')
plt.xlabel('Percentage of days not working')
plt.show()

x = mean_count_longrun
plt.hist(x, range = (int(min(x)-1), int(max(x)+1)), bins=(int(int(max(x)+1) - int(min(x)-1))), color='0.7')
plt.ylabel('Frequency')
plt.xlabel('Longest number of days worked in a row')
plt.show()

x = mean_count_manyHelps
plt.hist(x, range = (int(min(x)-1), int(max(x)+1)), bins=(int(int(max(x)+1) - int(min(x)-1))), color='0.7')
plt.ylabel('Frequency')
plt.xlabel('Number of days redistributed to critical area')
plt.show()
