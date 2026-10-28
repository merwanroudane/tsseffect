###############################################################################
# Reference output from the R package `tseffects`, for validating the Python port
#
# Run this in R, then the Python side reads the CSVs it writes and compares
# every estimate, standard error and confidence bound.
#
#   source("C:/Users/HP/Documents/xtpmg/tseffect/validate/r_reference.R")
#
# Needs: tseffects, sandwich, car  (mpoly and ggplot2 come in with tseffects)
###############################################################################

library(tseffects)
library(sandwich)

OUT <- "C:/Users/HP/Documents/xtpmg/tseffect/validate/r_out"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

data(toy.ts.interaction.data)
dat <- toy.ts.interaction.data

write.csv(dat, file.path(OUT, "toy_data.csv"), row.names = FALSE)

# ---------------------------------------------------------------- models ----
m.levels  <- lm(y ~ l_1_y + x + l_1_x, data = dat)
m.dx      <- lm(y ~ l_1_y + d_x + l_1_d_x, data = dat)
m.dy      <- lm(d_y ~ l_1_d_y + x + l_1_x, data = dat)
m.gecm    <- lm(d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x, data = dat)
m.gecm2   <- lm(d_y ~ l_1_y + l_1_d_y + l_2_d_y + l_1_x + d_x + l_1_d_x + l_2_d_x,
                data = dat)
m.inter   <- lm(y ~ l_1_y + x + l_1_x + z + l_1_z +
                  x_z + z_l_1_x + x_l_1_z + l_1_x_l_1_z, data = dat)
m.fdl     <- lm(y ~ x + l_1_x + l_2_x, data = dat)

models <- list(levels = m.levels, dx = m.dx, dy = m.dy, gecm = m.gecm,
               gecm2 = m.gecm2, inter = m.inter, fdl = m.fdl)

# Coefficients, so Python fits the identical model before comparing effects.
for (nm in names(models)) {
  co <- coef(models[[nm]])
  write.csv(data.frame(term = names(co), estimate = as.numeric(co)),
            file.path(OUT, paste0("coef_", nm, ".csv")), row.names = FALSE)
}

# ------------------------------------------------- vcovHC, every HC type ----
# This is the one piece the Python port reimplements from scratch (statsmodels
# ships only HC0-HC3), so every type gets dumped for an exact comparison.
hc.types <- c("const", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5")
for (nm in names(models)) {
  for (ty in hc.types) {
    V <- vcovHC(models[[nm]], type = ty)
    write.csv(as.data.frame(V),
              file.path(OUT, paste0("vcov_", nm, "_", ty, ".csv")),
              row.names = TRUE)
  }
}

# ------------------------------------------------------------- the cases ----
# Each entry writes <tag>.csv with the estimates and <tag>_formulae.txt with the
# algebra, so the port can be checked on both.
save.case <- function(tag, obj) {
  est <- obj$estimates
  write.csv(est, file.path(OUT, paste0(tag, ".csv")), row.names = FALSE)
  if (!is.null(obj$formulae)) {
    txt <- vapply(seq_along(obj$formulae), function(i) {
      paste0(names(obj$formulae)[i], " :: ",
             paste(unlist(obj$formulae[[i]]), collapse = " "))
    }, character(1))
    writeLines(txt, file.path(OUT, paste0(tag, "_formulae.txt")))
  }
  cat("wrote", tag, "-", nrow(est), "rows\n")
}

run <- function(tag, expr) {
  out <- tryCatch(suppressWarnings(eval(expr)),
                  error = function(e) { cat("FAILED", tag, ":", conditionMessage(e), "\n"); NULL })
  if (!is.null(out)) save.case(tag, out)
}

common <- list(return.data = TRUE, return.plot = FALSE, return.formulae = TRUE,
               s.limit = 10)

# --- A-D: ADL in levels, the four standard shock histories ---
run("A_adl_pulse", quote(GDRF.adl.plot(model = m.levels,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
  shock.history = "pulse", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("B_adl_step", quote(GDRF.adl.plot(model = m.levels,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
  shock.history = "step", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("C_adl_pulse_cumulative", quote(GDRF.adl.plot(model = m.levels,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
  shock.history = "pulse", cumulative = TRUE, s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("D_adl_h1", quote(GDRF.adl.plot(model = m.levels,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
  shock.history = 1, s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- E-F: x estimated in differences, read in levels and in differences ---
run("E_adl_dx_levels", quote(GDRF.adl.plot(model = m.dx,
  x.vrbl = c("d_x" = 0, "l_1_d_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 1, d.y = 0,
  shock.history = "pulse", inferences.x = "levels", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("F_adl_dx_differences", quote(GDRF.adl.plot(model = m.dx,
  x.vrbl = c("d_x" = 0, "l_1_d_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 1, d.y = 0,
  shock.history = "pulse", inferences.x = "differences", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- G-H: y estimated in differences, read in levels and in differences ---
run("G_adl_dy_levels", quote(GDRF.adl.plot(model = m.dy,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_d_y" = 1), d.x = 0, d.y = 1,
  shock.history = "pulse", inferences.y = "levels", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("H_adl_dy_differences", quote(GDRF.adl.plot(model = m.dy,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_d_y" = 1), d.x = 0, d.y = 1,
  shock.history = "pulse", inferences.y = "differences", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- I: a bespoke shock history (an interrupted trend) ---
run("I_adl_custom", quote(GDRF.adl.plot(model = m.levels,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
  shock.history = c(1, 2, 3, 0, 0, 4, 0, 0, 0, 1, 1), s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("I2_adl_custom_cumulative", quote(GDRF.adl.plot(model = m.levels,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
  shock.history = c(1, 2, 3, 0, 0, 4, 0, 0, 0, 1, 1), cumulative = TRUE,
  s.limit = 10, return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- J-K: fitted values, from a steady state and from an asserted baseline ---
run("J_adl_fitted_steadystate", quote(GDRF.adl.plot(model = m.levels,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
  shock.history = "pulse", effect.type = "fitted",
  prediction.values = list("x" = 1, "l_1_x" = 1),
  s.limit = 10, return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("K_adl_fitted_baseline", quote(GDRF.adl.plot(model = m.dy,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_d_y" = 1), d.x = 0, d.y = 1,
  shock.history = "pulse", effect.type = "fitted", baseline.y = 3,
  baseline.y.se = 2.5, shock.size = 2,
  s.limit = 10, return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- L-N: GECM ---
run("L_gecm_pulse", quote(GDRF.gecm.plot(model = m.gecm,
  x.vrbl = c("l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
  x.vrbl.d.x = 0, y.vrbl.d.y = 0,
  x.d.vrbl = c("d_x" = 0, "l_1_d_x" = 1), y.d.vrbl = c("l_1_d_y" = 1),
  x.d.vrbl.d.x = 1, y.d.vrbl.d.y = 1,
  shock.history = "pulse", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("M_gecm_step", quote(GDRF.gecm.plot(model = m.gecm,
  x.vrbl = c("l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
  x.vrbl.d.x = 0, y.vrbl.d.y = 0,
  x.d.vrbl = c("d_x" = 0, "l_1_d_x" = 1), y.d.vrbl = c("l_1_d_y" = 1),
  x.d.vrbl.d.x = 1, y.d.vrbl.d.y = 1,
  shock.history = "step", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("N_gecm_longer_lags", quote(GDRF.gecm.plot(model = m.gecm2,
  x.vrbl = c("l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
  x.vrbl.d.x = 0, y.vrbl.d.y = 0,
  x.d.vrbl = c("d_x" = 0, "l_1_d_x" = 1, "l_2_d_x" = 2),
  y.d.vrbl = c("l_1_d_y" = 1, "l_2_d_y" = 2),
  x.d.vrbl.d.x = 1, y.d.vrbl.d.y = 1,
  shock.history = "step", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- O: finite distributed lag, no lagged dependent variable ---
run("O_fdl_pulse", quote(GDRF.adl.plot(model = m.fdl,
  x.vrbl = c("x" = 0, "l_1_x" = 1, "l_2_x" = 2), y.vrbl = NULL, d.x = 0, d.y = 0,
  shock.history = "pulse", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- P-S: interactions ---
run("P_interact_zlines", quote(interact.adl.plot(model = m.inter,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
  z.vrbl = c("z" = 0, "l_1_z" = 1),
  x.z.vrbl = c("x_z" = 0, "z_l_1_x" = 1, "x_l_1_z" = 0, "l_1_x_l_1_z" = 1),
  z.vals = c(-1, 0, 1), effect.type = "discrete",
  plot.type = "lines", line.options = "z.lines", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("Q_interact_step", quote(interact.adl.plot(model = m.inter,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
  z.vrbl = c("z" = 0, "l_1_z" = 1),
  x.z.vrbl = c("x_z" = 0, "z_l_1_x" = 1, "x_l_1_z" = 0, "l_1_x_l_1_z" = 1),
  z.vals = c(-1, 0, 1), effect.type = "discrete", shock.history = "step",
  plot.type = "lines", line.options = "z.lines", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("R_interact_cumulative", quote(interact.adl.plot(model = m.inter,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
  z.vrbl = c("z" = 0, "l_1_z" = 1),
  x.z.vrbl = c("x_z" = 0, "z_l_1_x" = 1, "x_l_1_z" = 0, "l_1_x_l_1_z" = 1),
  z.vals = c(-1, 0, 1), effect.type = "discrete", shock.history = "impulse",
  cumulative = TRUE,
  plot.type = "lines", line.options = "z.lines", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

run("S_interact_fitted", quote(interact.adl.plot(model = m.inter,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
  z.vrbl = c("z" = 0, "l_1_z" = 1),
  x.z.vrbl = c("x_z" = 0, "z_l_1_x" = 1, "x_l_1_z" = 0, "l_1_x_l_1_z" = 1),
  z.vals = c(-1, 0, 1), effect.type = "fitted",
  prediction.values = list("x" = 1, "l_1_x" = 1,
                           "x_z" = 1, "z_l_1_x" = 1,
                           "x_l_1_z" = 1, "l_1_x_l_1_z" = 1),
  plot.type = "lines", line.options = "z.lines", s.limit = 10,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- T: robust standard errors, every HC type on one specification ---
for (ty in hc.types) {
  run(paste0("T_adl_pulse_", ty), bquote(GDRF.adl.plot(model = m.levels,
    x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
    shock.history = "step", se.type = .(ty), s.limit = 6,
    return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))
}

# --- U: a non-default confidence level ---
run("U_adl_step_level80", quote(GDRF.adl.plot(model = m.levels,
  x.vrbl = c("x" = 0, "l_1_x" = 1), y.vrbl = c("l_1_y" = 1), d.x = 0, d.y = 0,
  shock.history = "step", dM.level = 0.80, s.limit = 6,
  return.data = TRUE, return.plot = FALSE, return.formulae = TRUE)))

# --- V: the symbolic pieces on their own ---
pulses <- pulse.calculator(x.vrbl = c("x" = 0, "l_1_x" = 1),
                           y.vrbl = c("l_1_y" = 1), limit = 5)
writeLines(vapply(pulses, function(p) paste(capture.output(print(p, stars = TRUE)),
                                            collapse = " "), character(1)),
           file.path(OUT, "V_pulses.txt"))

for (cfg in list(c(0, 0, -1), c(0, 0, 0), c(1, 0, -1), c(0, 1, -1),
                 c(1, 0, 1), c(2, 0, -1), c(0, 2, -1))) {
  g <- general.calculator(d.x = cfg[1], d.y = cfg[2], h = cfg[3],
                          limit = 5, pulses = pulses)
  tag <- paste0("V_general_dx", cfg[1], "_dy", cfg[2], "_h", cfg[3])
  writeLines(unlist(g$formulae), file.path(OUT, paste0(tag, ".txt")))
  write.csv(do.call(rbind, lapply(seq_along(g$shocks), function(i) {
    data.frame(s = i - 1, j = seq_along(g$shocks[[i]]) - 1,
               weight = g$shocks[[i]])
  })), file.path(OUT, paste0(tag, "_shocks.csv")), row.names = FALSE)
}

adl.coef <- gecm.to.adl(x.vrbl = c("l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
                        x.d.vrbl = c("d_x" = 0, "l_1_d_x" = 1),
                        y.d.vrbl = c("l_1_d_y" = 1))
writeLines(c(paste("x:", paste(names(adl.coef$x.vrbl.adl), adl.coef$x.vrbl.adl,
                               sep = " = ", collapse = " | ")),
             paste("y:", paste(names(adl.coef$y.vrbl.adl), adl.coef$y.vrbl.adl,
                               sep = " = ", collapse = " | "))),
           file.path(OUT, "V_gecm_to_adl.txt"))

adl.coef2 <- gecm.to.adl(x.vrbl = c("l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
                         x.d.vrbl = c("d_x" = 0, "l_1_d_x" = 1, "l_2_d_x" = 2),
                         y.d.vrbl = c("l_1_d_y" = 1, "l_2_d_y" = 2))
writeLines(c(paste("x:", paste(names(adl.coef2$x.vrbl.adl), adl.coef2$x.vrbl.adl,
                               sep = " = ", collapse = " | ")),
             paste("y:", paste(names(adl.coef2$y.vrbl.adl), adl.coef2$y.vrbl.adl,
                               sep = " = ", collapse = " | "))),
           file.path(OUT, "V_gecm_to_adl_longer.txt"))

adl.coef3 <- gecm.to.adl(x.vrbl = c("l_1_x" = 1), y.vrbl = c("l_1_y" = 1),
                         x.d.vrbl = c("d_x" = 0), y.d.vrbl = NULL)
writeLines(c(paste("x:", paste(names(adl.coef3$x.vrbl.adl), adl.coef3$x.vrbl.adl,
                               sep = " = ", collapse = " | ")),
             paste("y:", paste(names(adl.coef3$y.vrbl.adl), adl.coef3$y.vrbl.adl,
                               sep = " = ", collapse = " | "))),
           file.path(OUT, "V_gecm_to_adl_minimal.txt"))

cat("\nsessionInfo:\n")
print(sessionInfo()$otherPkgs$tseffects$Version)
cat("\nAll reference output written to:", OUT, "\n")
