import plot_generator
import read_File

f = read_File.ReadFile("G_22.test_BIS")
f.read_stream_G()   #reads the binary G file

#Print the parameters
f.print_parameters()
