# plot bar graph for model accuracy
ggplot(data = model_accuracy, mapping = aes(x = Model.Name, y = Model.Accuracy)) +
  geom_bar(stat = "identity", fill = "#52d976", col = "#076921") +
  ggtitle("Model Accuracy") +
  geom_label(mapping = aes(label = round(Model.Accuracy, 2)))