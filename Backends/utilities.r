require("ggplot2")

# Construct the regularized antGLasso matrices
threshold.matrix <- function(mat., threshold) {
    mat <- matrix(0, dim(mat.)[[1]], dim(mat.)[[2]])
    mat[mat > 0] <- 0 # Only negative values (no substantial difference)
    mat[abs(mat.) < threshold] <- 0
    mat[abs(mat.) > threshold] <- 1
    diag(mat) <- 0
    return(mat)
}

num.elements <- function(mat) {
    sum(mat != 0)
}

get.assortativity.at.level <- function(adjacency.graph, taxmat, taxa.level) {
    groups <- as.integer(as.factor(taxmat[, taxa.level]))
    return(assortativity(adjacency.graph, groups))
}

get.assortativity.at.levels <- function(adjacency.graph, taxmat) {
    curried.assortativity <- function(taxa.level) get.assortativity.at.level(
        adjacency.graph,
        taxmat,
        taxa.level
    )
    return(
        lapply(
            colnames(taxmat),
            curried.assortativity
        )[2:length(colnames(taxmat))]
    )        
}

plot.all.assortativities <- function(path, taxmat, lambdas) {
    graphs <- lapply(
        lapply(path, graph.adjacency),
        as.undirected
    )
    assortativities <- lapply(
        graphs,
        function(graph) get.assortativity.at.levels(graph, taxmat)
    )

    # Remove first element as full of NaNs
    assortativities <- assortativities[2:length(assortativities)]
    lambdas.short <- lambdas[2:length(lambdas)]
    
    assortativities.1 <- as.numeric(lapply(assortativities, function(l) l[[1]]))
    assortativities.2 <- as.numeric(lapply(assortativities, function(l) l[[2]]))
    assortativities.3 <- as.numeric(lapply(assortativities, function(l) l[[3]]))
    assortativities.4 <- as.numeric(lapply(assortativities, function(l) l[[4]]))
    assortativities.5 <- as.numeric(lapply(assortativities, function(l) l[[5]]))
    ggplot(
        data.frame(assortativities.1),
        aes(x=lambdas.short)
    ) +
        geom_line(aes(y = assortativities.1, color = "Phylum")) +
        geom_line(aes(y = assortativities.2, color = "Class")) +
        geom_line(aes(y = assortativities.3, color = "Order")) +
        geom_line(aes(y = assortativities.4, color = "Family")) +
        geom_line(aes(y = assortativities.5, color = "Genus")) +
        scale_colour_manual("", 
            breaks = c("Phylum", "Class", "Order", "Family", "Genus"),
            values = c("black", "blue", "maroon", "red", "orange")
        ) +
        theme(legend.position = "top") +
        labs(x = "Regularization parameter lambda", y = "Assortativities") +
        ggtitle("Assortativities at different taxonomic levels")
}
                                           
plot.compared.assortativities <- function(
    paths,
    taxmat,
    line.names,
    title = "Assortativities at different taxonomic levels"
) {
    final.plot <- ggplot()
    min.edges <- 10000000
    max.edges <- 0
    for (i in 1:length(paths)) {
        path <- paths[[i]]
        graphs <- lapply(
            lapply(path, graph.adjacency),
            as.undirected
        )
        assortativities <- lapply(
            graphs,
            function(graph) get.assortativity.at.levels(graph, taxmat)
        )

        x.sparsity <- sapply(path, num.elements)
        min.edges <- min(min.edges, min(x.sparsity))
        max.edges <- max(max.edges, max(x.sparsity))

        assortativities.1 <- as.numeric(lapply(assortativities, function(l) l[[1]]))
        assortativities.2 <- as.numeric(lapply(assortativities, function(l) l[[2]]))
        assortativities.3 <- as.numeric(lapply(assortativities, function(l) l[[3]]))
        assortativities.4 <- as.numeric(lapply(assortativities, function(l) l[[4]]))
        assortativities.5 <- as.numeric(lapply(assortativities, function(l) l[[5]]))
              
        final.plot <- final.plot +
            geom_line(
                aes_string(
                    x=x.sparsity,
                    y=assortativities.1,
                    color=shQuote("Phylum"),
                    linetype=line.names[[i]]
                )
            ) +
            geom_line(
                aes_string(
                    x=x.sparsity,
                    y=assortativities.2,
                    color=shQuote("Class"),
                    linetype=line.names[[i]]
                )
            ) +
            geom_line(
                aes_string(
                    x=x.sparsity,
                    y=assortativities.3,
                    color=shQuote("Order"),
                    linetype=line.names[[i]]
                )
            ) +
            geom_line(
                aes_string(
                    x=x.sparsity,
                    y=assortativities.4,
                    color=shQuote("Family"),
                    linetype=line.names[[i]]
                )
            ) +
            geom_line(
                aes_string(
                    x=x.sparsity,
                    y=assortativities.5,
                    color=shQuote("Genus"),
                    linetype=line.names[[i]]
                )
            )
    }
                                           
    final.plot +
        scale_x_log10() + 
        scale_colour_manual("", 
            breaks = c("Phylum", "Class", "Order", "Family", "Genus"),
            values = c("black", "blue", "maroon", "red", "orange")
        ) +
        theme(legend.position = "top") +
        labs(x = "Number of Edges", y = "Assortativities") +
        ggtitle(title)# +
        #xlim(min.edges, max.edges)
}
                                               
                                               plot.graph <- function(adjacency.graph, taxmat_el, taxa.level) {
    color.graph.data <- color.graph(taxmat_el, taxa.level)
    taxmat.colors <- color.graph.data[[1]]
    unique.taxa <- color.graph.data[[2]]
    hex.map <- color.graph.data[[3]]
    plot(
        adjacency.graph,
        edge.color = "green",
        vertex.size = 5,
        vertex.label = "",
        vertex.color = taxmat.colors[,"color"],
        margin=c(0, 0, 0, 0)
    )
    plot(
        adjacency.graph,
        vertex.label.color = "#00000000",
        vertex.color = "#00000000",
        edge.color = "#00000000",
        vertex.frame.color = "#00000000"
    )
    legend('top',
           legend = unique.taxa,
           bg = "#757575",
           fill = hex.map,
           ncol = 2
    )
}

# Make a function to color the graph
color.graph <- function(taxmat_el, taxa.level) {
    # Add a "color" column to prepare for coloring graph nodes
    taxmat.colors <- cbind(taxmat_el, "#000000")
    colnames(taxmat.colors)[length(colnames(taxmat.colors))] <- "color"
    unique.taxa <- unique(taxmat.colors[,taxa.level])
    num.taxa <- length(unique.taxa)
    increment <- 89 / (num.taxa-1)
    increments <- 10 + round(increment * 0:(num.taxa-1))
    hex.map <- paste("#19", increments, "60", sep="")
    for (i in 1:num.taxa) {
        taxmat.colors[
            unique.taxa[i]==taxmat.colors[,taxa.level],
            "color"
        ] <- hex.map[i]
    }
    return(list(
        taxmat.colors,
        unique.taxa,
        hex.map
    ))
}