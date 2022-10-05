install.packages("mvtnorm")

library(readxl)
datos <- read_excel("Desktop/pibinflacion.xlsx")
head(datos)
length(datos$inflacion)

colnames(datos) <- c("PIB", "inflacion")

noerror_mod <- lm(PIB ~ inflacion, data = datos)

datos$inflacion <- datos$inflacion + rnorm(n = length(datos$inflacion), sd = sqrt(0.9))
error_mod <- lm(datos$PIB ~ datos$inflacion, data = datos)

noerror_mod$coefficients

error_mod$coefficients

plot(datos$inflacion, datos$PIB, 
     pch = 20, 
     col = "steelblue",
     xlab = "X",
     ylab = "Y")

abline(coef = c(15839079, -1800000), 
       col = "darkgreen",
       lwd  = 1.5)

abline(noerror_mod, 
       col = "purple",
       lwd  = 1.5)

abline(error_mod, 
       col = "darkred",
       lwd  = 1.5)

legend("topleft",
       bg = "transparent",
       cex = 0.8,
       lty = 1,
       col = c("darkgreen", "purple", "darkred"), 
       legend = c("Population", "No Errors", "Errors"))

#Obteniendo el valor estimado corregido con la varianza del vector de datos de inflacion, del sesgo añadido y del resultado de la regresión para los datos con error de medicion,se obtiene: 
((var(datos$inflacion) + 0.9) / var(datos$inflacion)) * -550942


