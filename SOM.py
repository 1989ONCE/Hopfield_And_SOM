from sklearn.model_selection import train_test_split
import numpy as np
import random, math, time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Unsupervised Learning
# Use the SOM algorithm to cluster the data

class SOM():
    def __init__(self, dataset_name, data, epoch_limit, lrn_rate, sigma):
        self.dataset_name = dataset_name
        self.data = np.array(data, dtype=np.float64)
        self.init_lrn_rate = lrn_rate
        self.lrn_rate = lrn_rate # initial learning rate
        self.epoch_limit = epoch_limit
        self.init_sigma = sigma # initial neighborhood radius
        self.sigma = sigma
        self.topology_dim = 2
        self.weight = np.array([])

        
        
        self.shuffle_data()
        self.input_dim = self.data.shape[1] - 1
        self.norm_input = self.normalize_whole_list(self.data[:, :-1])
        self.label = self.data[:, -1]

        # Minisom package recommended map size: 5 * sqrt(number of samples)
        # map_size represents the number of neurons in the map
        # map is a tuple that represents the number of neurons by row and column
        self.map_size = 5 * int(np.sqrt(len(self.norm_input)))
        self.map = (math.ceil(math.sqrt(self.map_size)), math.ceil(math.sqrt(self.map_size)))

        self.init_weight(self.input_dim)
        # τ：時間常數，控制衰減速度，通常為 total_epochs / log(sigma_0)。
        # Ensure that init_sigma > 1 to avoid log(<=0)
        if float(self.init_sigma) < 1:
            raise ValueError("init_sigma must be greater than 1 for a valid logarithm.")
        self.time_constant = self.epoch_limit / np.log(self.init_sigma)
        
    def shuffle_data(self):
        np.random.shuffle(self.data)
        # self.rand_data = np.array(self.data)
        #random.shuffle(self.data) 這個會錯很大
        

    def init_weight(self, dim):
        # self.weight = np.array([round(random.uniform(-1, 1), 2) for i in range(dim)])
        self.weight = np.random.uniform(
            low=0,
            high=1,
            size=(self.map[0], self.map[1], self.input_dim)
        )
        # print('Initial Weight:', self.weight)
        # print('Weight Shape:', self.weight.shape)

    def normalize_whole_list(self, x):
        self.mean = np.mean(x, axis=0, dtype=np.float64)
        self.std = np.std(x, axis=0, dtype=np.float64)
        # add epilson to avoid division by zero
        if self.std.any() == 0:
            self.std += np.finfo(np.float64).eps
        normalized_x = (x - self.mean) / self.std
        return normalized_x
    
    def denormalize_whole_list(self, x):
        return x * (self.std) + self.mean

    def train(self, update_callback=None):
        self.epoch = 0
        try:
            for epoch in range(1, self.epoch_limit + 1):
                init_time = time.time()
                self.update_sigma(epoch)
                self.update_lrn_rate(epoch)

                QE = 0
                for i in range(len(self.norm_input)):
                    x = self.norm_input[i]
                    bmu, bmu_index, bmu_dist = self.find_best_matching_unit(x)
                    QE += bmu_dist
                    for i in range(self.map[0]):
                        for j in range(self.map[1]):
                            dist_to_bmu = self.euclidean_distance(bmu_index, np.array([i, j]))
                            if dist_to_bmu <= self.sigma:
                                self.weight[i][j] += self.lrn_rate * self.neighborhood_influence(dist_to_bmu, self.sigma) * (x - self.weight[i][j])

                QE /= len(self.norm_input)
                process_time = time.time() - init_time
                fig, fig2 = self.umatrix_visualize(epoch, QE, process_time, self.sigma, self.lrn_rate)
                
                # Call the callback function to update the GUI
                if update_callback:
                    update_callback(fig, fig2)

                print(f'Epoch {epoch} completed. Process Time: {process_time:.5f}s, QE: {QE:.4f}')
            print('Training is completely done!')
            return True
        except KeyboardInterrupt:
            print('Training interrupted by user.')
            return True


    def find_best_matching_unit(self, x):
        min_dist = float('inf')
        bmu = None
        bmu_index = None

        for i in range(self.weight.shape[0]):
            for j in range(self.weight.shape[1]):
                w = self.weight[i][j]
                dist = self.euclidean_distance(x, w)
            
                if dist < min_dist:
                    min_dist = dist
                    bmu = w
                    bmu_index = np.array([i, j])
        return bmu, bmu_index, min_dist

    def euclidean_distance(self, x, w):
        x_w = x - w
        val = 0
        for i in range(len(x_w)):
            val += x_w[i]**2
        
        dist = math.sqrt(val)
        return dist
    
    def update_lrn_rate(self, epoch):
        self.lrn_rate = self.init_lrn_rate * np.exp(-epoch / self.time_constant)
    def update_sigma(self, epoch):
        self.sigma = self.init_sigma * np.exp(-epoch / self.time_constant)

    def neighborhood_influence(self, dist, sigma):
        return np.exp(- (dist**2) / (2 * (sigma**2)))

    def get_map_size(self):
        return self.map

    def umatrix_visualize(self, epoch, qe, process_time, sigma, lrn_rate):
        u_matrix = self.calculate_u_matrix()
        fig, ax = plt.subplots(figsize=(7, 7))  # Adjusted figure size for clarity
        fig2, ax2 = plt.subplots(figsize=(4, 4))  # fig for sigma and lrn_rate changes over time

        # Plotting the U-Matrix
        cax = ax.imshow(u_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(cax, ax=ax, orientation='vertical', label='Distance')
        ax.set_title(f'Dataset: {self.dataset_name}\nU-Matrix at Epoch {epoch}\nQuantization Error: {qe:.4f}, Time: {process_time:.5f}s \nSigma: {sigma:.4f}, Learning Rate: {lrn_rate:.4f}')
        
        # Adding axis labels to clarify the neuron grid
        ax.set_xlabel('Neuron Index (X)', fontsize=10)
        ax.set_ylabel('Neuron Index (Y)', fontsize=10)

        colors = list(mcolors.TABLEAU_COLORS.values())  # Use Tableau colors for distinct classes
        for cnt, x in enumerate(self.norm_input):
            _, winner_idx, _ = self.find_best_matching_unit(x)
            label = int(self.label[cnt])  # Assuming label is already an integer class
            ax.text(
                winner_idx[1],  # x-coordinate of neuron square
                winner_idx[0],  # y-coordinate of neuron square
                str(label),  # Text to display (the class label)
                color=colors[label % len(colors)],
                fontsize=10,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # Add background for better visibility
            )

        # plot sigma and lrn_rate changes over time
        ax2.plot(range(1, epoch + 1), [self.init_sigma * np.exp(-i / self.time_constant) for i in range(1, epoch + 1)], label='Sigma')
        ax2.plot(range(1, epoch + 1), [self.init_lrn_rate * np.exp(-i / self.time_constant) for i in range(1, epoch + 1)], label='Learning Rate')
        ax2.set_title('Sigma and Learning Rate Decay over Time', fontsize=10)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid()
        
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        return fig, fig2

    # U-Matrix: Unified Distance Matrix
    # 顯示神經元之間的距離或相似性，幫助識別聚類結構。
    # 通常用色階，淺色代表距離小（神經元接近，可能屬於同一類別），深色代表距離大（神經元遠，可能屬於不同類別）
    def calculate_u_matrix(self):
        """Calculates the U-matrix for visualization."""
        u_matrix = np.zeros((self.map[0], self.map[1]))
        for i in range(self.map[0]):
            for j in range(self.map[1]):
                distances = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.map[0] and 0 <= nj < self.map[1] and (di != 0 or dj != 0):
                            neighbor_distance = self.euclidean_distance(self.weight[i, j], self.weight[ni, nj])
                            distances.append(neighbor_distance)
                if distances:
                    u_matrix[i, j] = np.mean(distances)
        return u_matrix