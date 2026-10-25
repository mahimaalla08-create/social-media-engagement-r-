
  
  
  # Install required packages
  install.packages(c("ggplot2", "dplyr", "randomForest", "arules"))
  
  # Load required libraries
  library(ggplot2)
  library(dplyr)
  library(randomForest)
  library(arules)
  
  # Load the CSV file
  data <- read.csv("sentimentdataset.csv")
  
  # Display first six rows
  head(data)
  
  # Display dataset structure
  str(data)
  
  # Display summary statistics
  summary(data)
  
  # Remove duplicate rows
  data <- unique(data)
  
  # Clean column names
  names(data) <- trimws(names(data))
  
  # Remove extra spaces from text values
  data <- data %>%
    mutate(across(where(is.character), trimws))
  
  # Convert likes and retweets to numeric
  data$Likes <- as.numeric(data$Likes)
  data$Retweets <- as.numeric(data$Retweets)
  
  # Remove missing likes and retweets
  data <- data %>%
    filter(!is.na(Likes), !is.na(Retweets))
  
  # Display cleaned dataset
  head(data)
  
  # Calculate total engagement
  data$engagement <- data$Likes + data$Retweets
  
  # Classify posts as high or low engagement
  data$high_engagement <- ifelse(
    data$engagement > median(data$engagement),
    1, 0
  )
  
  # Count hashtags in each post
  data$hashtag_count <- sapply(
    strsplit(as.character(data$Hashtags), "\\s+"),
    function(x) sum(grepl("#", x))
  )
  
  # Display new variables
  head(data[, c("Likes", "Retweets",
                "engagement", "hashtag_count")])
  
  # Compare engagement across platforms
  ggplot(data, aes(x = Platform, y = engagement,
                   fill = Platform)) +
    geom_boxplot() +
    labs(title = "Engagement by Platform",
         x = "Platform", y = "Likes + Retweets") +
    theme_minimal()
  
  # Compare engagement across sentiments
  ggplot(data, aes(x = Sentiment, y = engagement,
                   fill = Sentiment)) +
    geom_boxplot() +
    labs(title = "Engagement by Sentiment",
         x = "Sentiment", y = "Engagement") +
    theme_minimal() +
    theme(axis.text.x = element_text(
      angle = 45, hjust = 1
    ))
  
  # Analyze engagement by posting hour
  ggplot(data, aes(x = Hour, y = engagement)) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(title = "Engagement by Posting Hour",
         x = "Hour", y = "Engagement") +
    theme_minimal()
  
  # Compare likes and retweets
  ggplot(data, aes(x = Likes, y = Retweets)) +
    geom_point() +
    labs(title = "Likes vs Retweets",
         x = "Likes", y = "Retweets") +
    theme_minimal()
  
  # Convert sentiment to character
  data$Sentiment <- as.character(data$Sentiment)
  
  # Group sentiments into three categories
  data$Sentiment_Group <- ifelse(
    grepl("anger|heartache|lostlove|sad|fear|hate",
          data$Sentiment, ignore.case = TRUE),
    "Negative",
    ifelse(
      grepl("joy|positive|happy|optimism|success|confidence|romance",
            data$Sentiment, ignore.case = TRUE),
      "Positive",
      "Neutral"
    )
  )
  
  # Prepare model dataset
  model_data <- data.frame(
    engagement = data$engagement,
    Platform = trimws(data$Platform),
    Sentiment_Group = data$Sentiment_Group,
    Hour = as.numeric(data$Hour),
    hashtag_count = data$hashtag_count
  )
  
  # Remove missing and invalid values
  model_data <- model_data[
    complete.cases(model_data) &
      is.finite(model_data$engagement) &
      is.finite(model_data$Hour) &
      is.finite(model_data$hashtag_count),
  ]
  
  # Convert categorical variables to factors
  model_data$Platform <- factor(model_data$Platform)
  model_data$Sentiment_Group <- factor(
    model_data$Sentiment_Group
  )
  
  # Split data into training and testing sets
  set.seed(123)
  
  train_rows <- sample(
    seq_len(nrow(model_data)),
    floor(0.8 * nrow(model_data))
  )
  
  train <- model_data[train_rows, ]
  test <- model_data[-train_rows, ]
  
  # Build Linear Regression model
  model_lm <- lm(
    engagement ~ Platform + Sentiment_Group +
      Hour + hashtag_count,
    data = train
  )
  
  # Predict engagement
  pred <- predict(model_lm, newdata = test)
  actual <- test$engagement
  
  # Check for valid predictions
  valid <- is.finite(pred) & is.finite(actual)
  
  pred <- pred[valid]
  actual <- actual[valid]
  
  # Calculate evaluation metrics
  if (length(actual) >= 2) {
    
    # Root Mean Squared Error
    rmse <- sqrt(mean((actual - pred)^2))
    
    # Mean Absolute Error
    mae <- mean(abs(actual - pred))
    
    # R-squared
    r2 <- if (sum((actual - mean(actual))^2) > 0) {
      1 - sum((actual - pred)^2) /
        sum((actual - mean(actual))^2)
    } else {
      NA
    }
    
    # Display results
    cat("RMSE:", rmse, "\n")
    cat("MAE:", mae, "\n")
    cat("R-squared:", r2, "\n")
    
  } else {
    cat("Not enough valid predictions to calculate metrics.\n")
  }
  
  # Check model results
  cat("Total rows:", nrow(model_data), "\n")
  cat("Training rows:", nrow(train), "\n")
  cat("Testing rows:", nrow(test), "\n")
  cat("Valid predictions:", length(pred), "\n")
  
  # Display model summary
  summary(model_lm)
  
  #Random Forest Regression
  set.seed(123)
  
  model_rf <- randomForest(
    engagement ~ Platform + Sentiment_Group +
      Hour + hashtag_count,
    data = train,
    ntree = 100
  )
  
  pred_rf <- predict(model_rf, test)
  
  rmse_rf <- sqrt(mean((test$engagement - pred_rf)^2))
  mae_rf <- mean(abs(test$engagement - pred_rf))
  
  r2_rf <- 1 - sum((test$engagement - pred_rf)^2) /
    sum((test$engagement - mean(test$engagement))^2)
  
  cat("Random Forest RMSE:", rmse_rf, "\n")
  cat("Random Forest MAE:", mae_rf, "\n")
  cat("Random Forest R-squared:", r2_rf, "\n")
  
  #Logistic Regression
  # Create high engagement variable
  train$high_engagement <- ifelse(
    train$engagement > median(train$engagement),
    1, 0
  )
  
  test$high_engagement <- ifelse(
    test$engagement > median(train$engagement),
    1, 0
  )
  
  # Build logistic regression model
  model_log <- glm(
    high_engagement ~ Platform + Sentiment_Group +
      Hour + hashtag_count,
    data = train,
    family = binomial
  )
  
  # Predict probabilities
  prob <- predict(model_log, newdata = test, type = "response")
  
  # Convert probabilities into classes
  pred_class <- ifelse(prob >= 0.5, 1, 0)
  
  # Calculate accuracy
  actual <- test$high_engagement
  accuracy <- mean(pred_class == actual)
  
  cat("Logistic Regression Accuracy:",
      round(accuracy * 100, 2), "%\n")
  
  #Prepare Hashtags for Apriori
  tags <- as.character(data$Hashtags)
  
  baskets <- strsplit(tags, "\\s+")
  
  baskets <- lapply(baskets, function(x) {
    unique(x[grepl("#", x, fixed = TRUE)])
  })
  
  baskets <- baskets[lengths(baskets) >= 2]
  
  trans <- as(baskets, "transactions")
  
  summary(trans)
  data$Sentiment_Group <- ifelse(
    grepl("happ|joy|love|positive|happy|excit|gratitude|hope|admiration",
          tolower(data$Sentiment)), "Positive",
    ifelse(
      grepl("anger|sad|fear|hate|negative|disappoint|anxious|grief|frustrat",
            tolower(data$Sentiment)), "Negative",
      "Neutral"
    )
  )
  
  table(data$Sentiment_Group)
  head(data[, c("Text", "Sentiment", "Sentiment_Group")], 10)
  # Apply Apriori Association Mining
  rules <- apriori(
    trans,
    parameter = list(
      supp = 0.01,
      conf = 0.5,
      minlen = 2
    )
  )
  
  rules <- sort(rules, by = "lift", decreasing = TRUE)
  
  inspect(head(rules, 10))
  library(arules)
  
  tags <- as.character(data$Hashtags)
  baskets <- strsplit(tags, "\\s+")
  
  baskets <- lapply(baskets, function(x) {
    unique(x[grepl("#", x, fixed = TRUE)])
  })
  
  baskets <- baskets[lengths(baskets) >= 2]
  
  trans <- as(baskets, "transactions")
  
  summary(trans)
  rules <- apriori(
    trans,
    parameter = list(
      supp = 0.01,
      conf = 0.5,
      minlen = 2
    )
  )
  
  rules <- sort(rules, by = "lift", decreasing = TRUE)
  
  inspect(head(rules, 10))
  library(arulesViz)
  
  
  #Compare Prediction Models
  results <- data.frame(
    Model = c("Linear Regression", "Random Forest"),
    RMSE = c(rmse, rmse_rf),
    MAE = c(mae, mae_rf),
    R_Squared = c(r2, r2_rf)
  )
  
  print(results)
  
