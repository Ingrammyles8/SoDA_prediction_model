new_coef <- read.csv('/Users/ashleyzahabian/Desktop/ridge_regression_coef.csv', header = TRUE)

 # Simple horizontal bar plot with new coefficients and significance 

 ggplot(new_coef, aes(reorder(features, Coefficients), Coefficients)) + 
   geom_col(aes(fill = Coefficients > 0)) + coord_flip() + 
   labs(x = "Features", y = "Beta Estimate") +  
   guides(fill = FALSE) +
   ggtitle("Ridge Regression Beta Estimates for SoDA Model") +
   theme(axis.line = element_line(color = "black"),
         axis.text = element_text(size = 9),
         plot.title = element_text(size = 14, face = "bold"),
         axis.title.x = element_text(size = 10, face = "bold"),
         axis.title.y = element_text(size = 10, face = "bold"),
         legend.position = "none") 

 