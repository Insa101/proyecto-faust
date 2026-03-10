dir.base = "~/Dropbox/MasterD/CursoBigData/04Comunicacion/Ejercicios"
clientes <- read.csv(paste0(dir.base,"/datos_clientes.csv"),sep=";")


fechas_ini_fact = seq(as.Date("2016-01-01"),as.Date("2016-12-01"),by="month")
fechas_fin_fact = c(fechas_ini_fact[-1] -1,as.Date("2016-12-31"))

for(i in 1:nrow(clientes))
  for(j in 1:length(fechas_ini_fact)){
    cliente=as.list(clientes[i,])
    inicio_factura = as.character(fechas_ini_fact[j])
    fin_factura = as.character(fechas_fin_fact[j])
    rmarkdown::render("~/Dropbox/MasterD/CursoBigData/04Comunicacion/Ejercicios/Ej_U04_E02_plantilla.Rmd",
                    output_file = paste0("facturas/factura_cli_",cliente$id,"_",format(as.Date(inicio_factura),"%b%Y"),".html"), 
                                         params=list(cliente=cliente,inicio_factura=inicio_factura,fin_factura=fin_factura))
}

