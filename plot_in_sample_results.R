if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot,
  sf,
  maps,
  ggthemes,
  cowplot,
  scico
)


# Read data ---------------------------------------------------------------

test_kge <- read_csv("./data/test_scores_KGEs_test.csv", col_names = "KGE", show_col_types = FALSE)


# Plot --------------------------------------------------------------------
data_plot <- test_kge
ggplot(data_plot, aes(KGE))+
  geom_histogram(bins = 50) +
  geom_vline(xintercept = median(test_kge$KGE),      # Add line for median
             col = "red",
             lwd = 2) +
  annotate("text",
           x = 0.65,
           y = 100,
           label = paste("Median =", round(median(test_kge$KGE), 3)),
           col = "red",
           size = 6)+
  theme_bw()+
  labs(x = "Test KGE")



