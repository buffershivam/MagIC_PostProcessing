import plot_generator
import read_File

f = read_File.ReadFile("G_22.test_BIS")
f.read_stream_G()   #reads the binary G file

#Print the parameters
f.print_parameters()

#Create plots of the G File
p = plot_generator.GeneratePlots(f)

p.generate_plot(prop="entropy",plot_type="Eq")
p.generate_plot(prop="vphi",plot_type="Ortho")
p.generate_plot(prop="vphi",plot_type="Mol")