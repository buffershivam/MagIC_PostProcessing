import plot_generator
import read_File
import budgets

f = read_File.ReadFile("G_100.test")
f.read_stream_G()   #reads the binary G file

#Print the parameters
# f.print_parameters()

b = budgets.GenerateBudgets(f)

Bflux_mean = b.meanBuoyFlux()
Diss_mean = b.meanDiss()
#Create plots of the G File
p = plot_generator.GeneratePlots(f)


Bflux_minus_Diss = Bflux_mean - Diss_mean

p.generate_meridional_plot(Bflux_minus_Diss,"Prod")
#p.generate_plot(prop="entropy",plot_type="Eq")
# p.generate_plot(prop="vphi",plot_type="Ortho")
#p.generate_plot(prop="vphi",plot_type="Mol")