def move_from_pc():
    socket_open("192.168.0.104", 30004, "socket_0")

    vals = socket_read_ascii_float(6, "socket_0", 2.0)

    x = vals[1]
    y = vals[2]
    z = vals[3]
    RX = vals[4]
    RY = vals[5]
    RZ = vals[6]

    movel(p[x-0.025, y-0.02, z-0.005, RX, RY, RZ], a=0.1, v=0.1)

    socket_close("socket_0")
end
move_from_pc()