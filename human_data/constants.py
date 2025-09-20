import numpy as np 

# forward-backward, left-right, up-down
# xyz
# camera standard coordinate: z-forward, x-right, y-down

xfyrzd2standard = np.array([
    [0, 1, 0, 0], 
    [0, 0, 1, 0], 
    [1, 0, 0, 0], 
    [0, 0, 0, 1]
], dtype=np.float32)

yfxrzu2standard = np.array([
    [1, 0, 0, 0], 
    [0, 0, 1, 0], 
    [0, -1, 0, 0], 
    [0, 0, 0, 1]
], dtype=np.float32)

standard2xfylzu = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

xfylzu2standard = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

zfyrxu2standard = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

ybzlxd2standard = np.array([
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

# xfzryu2zfylxd = np.array([
#     [0, -1, 0, 0],
#     [0, 0, -1, 0],
#     [1, 0, 0, 0]
# ])

