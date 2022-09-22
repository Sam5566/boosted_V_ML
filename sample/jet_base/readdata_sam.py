import ROOT as r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

obss = ['Jet.PT']

f = r.TFile.Open('test3/Events/run_4/tag_1_delphes_events.root', "READ")
#print (f.ls())

tree = f.Get ("Delphes")

for b in tree.GetListOfBranches():
    print ("branch :", b.GetName())

for entryNum in range (0 , 3):
    jet_PT = getattr(tree,"Jet.PT")
    print (jet_PT)

values = []
N = tree.GetEntries()
for entryNum in range (0 , 3):
    #print (entryNum)
    values.append([])
    for BranchName in range(len(obss)):
        #obs = tree.GetBranch(obss[BranchName])
        #obsName = obs.GetName()
        
        #print (obsName)
        tree.GetEntry(entryNum)
        jet = tree.Jet
        print (jet)
        print (f.GetLeaf("Jet.PT").GetValue(0))
        values[entryNum].append(tree.Jet)

    

values = np.array(values)
print ('start plotting')
plt.hist(values[:,0], bins=500)
plt.savefig('figures/Jet_PT.png', dpi=300)
