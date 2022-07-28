import uproot
import os
import awkward as awk
from VertexImageCreatorFunctions import createVertexImage, Observe_N_Vertices, CountTrackQuadrants, PlotQuadrantBars,\
    plotVertexThetaHistogram, plotTkThetaHistogram, plotVtxX_VtxY, plot2DVtxX_VtxY, files_from_directory_to_list, obtain_max_data_for_directory

b'Import ROOT files'
beamspot_coordinates = [-0.02479, 0.06929, 0.7899] #X, Y, Z
image_filepath = os.path.join(os.path.expanduser('~'), 'Box Sync', 'Neu-work', 'Longlive master', 'ML Vertex Images')
root_directory = os.path.join(os.path.expanduser('~'), 'Box Sync', 'Neu-work', 'Longlive master', 'roots', 'vertextree')

signalfiles, bkgfiles = files_from_directory_to_list(root_directory)
maxdict = obtain_max_data_for_directory(signalfiles, bkgfiles)
dim=256

b'Delete files from directory list if necessary'
#del(signalfiles[:3])
del(bkgfiles[:1])


b'Iteratively create images for signal and background using singular processing'
#for file in signalfiles:
#    print(file, flush=True)
#    btree = None
#    sroot = uproot.open(file)
#    stree = sroot['mfvVertexTreer']['tree_DV']
#    Observe_N_Vertices(1, stree, btree, beamspot_coordinates, w=dim, h=dim, ImageFilePath=image_filepath, showTitle=True,
#                       drawCenter=True, maxData=maxdict, rootFileName=file, IsSignal=True)
#    createVertexImage(stree, btree, beamspot_coordinates, event=0, vertex=0, w=256, h=256, ImageFilePath=image_filepath,
#                      IsSignal=True, showTitle=True, drawCenter=True, maxData=maxdict, rootFileName=file)

#for file in bkgfiles:
#    print(file)
#    stree = None
#    broot = uproot.open(file)
#    btree = broot['mfvVertexTreer']['tree_DV']
#    Observe_N_Vertices(1, stree, btree, beamspot_coordinates, w=dim, h=dim, ImageFilePath=image_filepath, showTitle=True,
#                       drawCenter=True, maxData=maxdict, rootFileName=file, IsSignal=False)
#    createVertexImage(stree, btree, beamspot_coordinates, event=0, vertex=0, w=256, h=256, ImageFilePath=image_filepath,
#                      IsSignal=False, showTitle=True, drawCenter=True, maxData=maxdict, rootFileName=file)

b'Iteratively create images for signal and background using multiprocessing'
if __name__ == '__main__':
#    for file in signalfiles:
#        print(file, flush=True)
#        btree = None
#        sroot = uproot.open(file)
#        stree = sroot['mfvVertexTreer']['tree_DV']
#        if file == signalfiles[0]:
#            start = 5734
#        else:
#            start = 0
#        Observe_N_Vertices('all', stree, btree, beamspot_coordinates, w=dim, h=dim, IsSignal=True, ImageFilePath=image_filepath,
#                           showTitle=False, drawCenter=False, maxData=maxdict, rootFileName=file, parallel=True, start=start)

    for file in bkgfiles:
        print(file, flush=True)
        stree = None
        broot = uproot.open(file)
        btree = broot['mfvVertexTreer']['tree_DV']
        if file == bkgfiles[0]:
            start = 0
            N = 20000
        else:
            start = 0
            N = 'all'
        Observe_N_Vertices(N, stree, btree, beamspot_coordinates, w=dim, h=dim, IsSignal=False, ImageFilePath=image_filepath,
                           showTitle=False, drawCenter=False, maxData=maxdict, rootFileName=file, parallel=True, start=start)

b'Run Individual Vertex'
#createVertexImage(stree, btree, beamspot_coordinates, event=0, vertex=0, w=32, h=32, ImageFilePath=image_filepath,
#                  IsSignal=True, showTitle=False, drawCenter=False)

b'Quadrant Analysis'
#sroot = uproot.open(signalfiles)
#broot = uproot.open(bkgfiles)
#stree = sroot['mfvVertexTreer']['tree_DV']
#btree = broot['mfvVertexTreer']['tree_DV']
#signal_quadrants = CountTrackQuadrants(stree, btree, beamspot_coordinates, IsSignal=True)
#background_quadrants = CountTrackQuadrants(stree, btree, beamspot_coordinates, IsSignal=False)
#PlotQuadrantBars(signal_quadrants, background_quadrants)

b'Vertex angle, track angle, and vertex position analyses'
#plotVertexThetaHistogram(stree, btree, beamspot_coordinates)
#plotTkThetaHistogram(stree, btree, beamspot_coordinates, rotation_constant=180, addVtxTheta=True, printInformation=False)
#plotVtxX_VtxY(stree, btree, beamspot_coordinates)
#plot2DVtxX_VtxY(stree, btree, beamspot_coordinates)