library(tidyverse)

plot_mosaic <- function(res, lineage1, lineage2, genome_length = 29903, sample = NULL) {
  filtered <- res %>%
    filter(recomb == 1, Lineage_1 == lineage1, Lineage_2 == lineage2) 
  
  mosaic_df <- filtered %>%
    rowwise() %>%
    mutate(
      lineage_path_vec = list(str_split(lineage_path, ";")[[1]]),
      breakpoint_vec = list(if (!is.na(breakpoints)) as.integer(str_split(breakpoints, ";")[[1]]) else integer(0))
    ) %>%
    mutate(
      blocks = list({
        bps <- c(1, breakpoint_vec + 1, genome_length + 1)
        path <- lineage_path_vec
        if (length(path) == 1) path <- rep(path, length(bps) - 1)
        tibble(start = bps[-length(bps)] - 1, end = bps[-1] - 1, lineage = path)
      })
    ) %>%
    unnest(blocks) %>%
    ungroup()
  
  # If sample is set, subsample unique sequences
  if (!is.null(sample)) {
    sampled_ids <- mosaic_df %>%
      distinct(sequence_id) %>%
      slice_sample(n = min(sample, nrow(.))) %>%
      pull(sequence_id)
    
    mosaic_df <- mosaic_df %>% filter(sequence_id %in% sampled_ids)
  }
  
  mosaic_df <- mosaic_df %>%
    mutate(seq_index = as.integer(factor(sequence_id, levels = unique(sequence_id))),
           start = ifelse(start == 0, 1, start),
           end = ifelse(end == 8, 9, end))
  
  ggplot(mosaic_df, aes(xmin = start, xmax = end, ymin = seq_index - 0.4, ymax = seq_index + 0.4, fill = lineage)) +
    geom_rect(color = "white") +
    scale_y_continuous(
      breaks = mosaic_df$seq_index,
      labels = mosaic_df$sequence_id,
      expand = expansion(add = 1)
    ) +
    labs(
      x = "Genome Position",
      y = "Sequence ID",
      fill = "Lineage",
      title = paste("Inferred Mosaic: ", lineage1, " + ", lineage2)
    ) +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 6))
}

get_mean_entropy <- function(res, lineage1, lineage2, genome_length = 29903) {
  # Step 1: Filter to only recombinants with *exactly* the two lineages
  filtered <- res %>%
    filter(recomb == 1, Lineage_1 == lineage1, Lineage_2 == lineage2) 
  
  if (nrow(filtered) == 0) {
    message("No recombinants found with exactly: ", lineage1, " and ", lineage2)
    return(NA_real_)
  }
  
  # Step 2: Convert each recombinant into a lineage vector
  get_lineage_vector <- function(breakpoints_str, path_str, genome_length) {
    breakpoints <- if (!is.na(breakpoints_str)) as.integer(str_split(breakpoints_str, ";")[[1]]) else integer(0)
    path <- str_split(path_str, ";")[[1]]
    
    starts <- c(1, breakpoints)
    ends <- c(breakpoints - 1, genome_length)
    
    lineage_vector <- rep(NA_character_, genome_length)
    for (i in seq_along(path)) {
      lineage_vector[starts[i]:ends[i]] <- path[i]
    }
    return(lineage_vector)
  }
  
  # Step 3: Apply to all rows to create lineage matrix
  lineage_mat <- filtered %>%
    rowwise() %>%
    mutate(lineage_vec = list(get_lineage_vector(breakpoints, lineage_path, genome_length))) %>%
    ungroup() %>%
    pull(lineage_vec) %>%
    do.call(rbind, .)
  
  # Step 4: Compute entropy at each position
  entropy_vec <- apply(lineage_mat, 2, function(col) {
    p <- prop.table(table(col))
    -sum(p * log2(p))
  })
  
  mean_entropy <- mean(entropy_vec, na.rm = TRUE)
  return(mean_entropy)
}
