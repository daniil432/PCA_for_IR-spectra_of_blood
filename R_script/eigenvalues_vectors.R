args <- commandArgs(trailingOnly = TRUE)
curdir <- args[1]

c_matrix <- as.matrix(read.table(paste0(curdir, "\\R_script\\input\\c_matrix.csv"), header=FALSE, sep=","))
b_matrix <- as.matrix(read.table(paste0(curdir, "\\R_script\\input\\b_matrix.csv"), header=FALSE, sep=","))

eigenvalues_vectors_c <- eigen(c_matrix)
eigenvalues_vectors_b <- eigen(b_matrix)

eigenvalues_c <- eigenvalues_vectors_c$values
eigenvectors_c <- t(eigenvalues_vectors_c$vectors)
eigenvalues_b <- eigenvalues_vectors_b$values
eigenvectors_b <- t(eigenvalues_vectors_b$vectors)

write.table(eigenvalues_c, paste0(curdir, "\\R_script\\output\\eigenvalues_c.csv"), quote=FALSE, sep=",", row.names=FALSE, col.names=FALSE)
write.table(eigenvectors_c, paste0(curdir, "\\R_script\\output\\eigenvectors_c.csv"), quote=FALSE, sep=",", row.names=FALSE, col.names=FALSE)
write.table(eigenvalues_b, paste0(curdir, "\\R_script\\output\\eigenvalues_b.csv"), quote=FALSE, sep=",", row.names=FALSE, col.names=FALSE)
write.table(eigenvectors_b, paste0(curdir, "\\R_script\\output\\eigenvectors_b.csv"), quote=FALSE, sep=",", row.names=FALSE, col.names=FALSE)