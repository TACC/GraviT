import os

def xyzzy(name, colors):
	print name
	f = open(name, "w")
	f.write("%d\n" % len(colors))
	for i,c in enumerate(colors):
		f.write("%f %f %f %f\n" % tuple([i / (len(colors)-1.0)] + c))
	f.close()

colors = []
colors.append([0         , 0           , 0.562493   ])
colors.append([0         , 0           , 1          ])
colors.append([0         , 1           , 1          ])
colors.append([0.500008  , 1           , 0.500008   ])
colors.append([1         , 1           , 0          ])
colors.append([1         , 0           , 0          ])
colors.append([0.500008  , 0           , 0          ])

xyzzy(os.environ["HOME"] + "/colormaps/Jet.cmap", colors)

colors = []
colors.append([0         , 0           , 0           ])
colors.append([0         , 0.120394    , 0.302678    ])
colors.append([0         , 0.216587    , 0.524575    ])
colors.append([0.0552529 , 0.345022    , 0.659495    ])
colors.append([0.128054  , 0.492592    , 0.720287    ])
colors.append([0.188952  , 0.641306    , 0.792096    ])
colors.append([0.327672  , 0.784939    , 0.873426    ])
colors.append([0.60824   , 0.892164    , 0.935546    ])
colors.append([0.881376  , 0.912184    , 0.818097    ])
colors.append([0.9514    , 0.835615    , 0.449271    ])
colors.append([0.904479  , 0.690486    , 0           ])
colors.append([0.854063  , 0.510857    , 0           ])
colors.append([0.777096  , 0.330175    , 0.000885023 ])
colors.append([0.672862  , 0.139086    , 0.00270085  ])
colors.append([0.508812  , 0           , 0           ])
colors.append([0.299413  , 0.000366217 , 0.000549325 ])
colors.append([0.0157473 , 0.00332647  , 0           ])

xyzzy(os.environ["HOME"] + "/colormaps/IceFire.cmap", colors)

colors = []
colors.append([0.231373  , 0.298039    , 0.752941    ])
colors.append([0.865003  , 0.865003    , 0.865003    ])
colors.append([0.705882  , 0.0156863   , 0.14902     ])

xyzzy(os.environ["HOME"] + "/colormaps/CoolWarm.cmap", colors)

colors = []
colors.append([0         , 0           , 1           ])
colors.append([1         , 0           , 0           ])

xyzzy(os.environ["HOME"] + "/colormaps/Rainbow.cmap", colors)

colors = []
colors.append([0., 0., 0.])
colors.append([1., 1., 1.])

xyzzy(os.environ["HOME"] + "/colormaps/Grayscale.cmap", colors)
