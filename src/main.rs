use rand::Rng;
use std::{error::Error, io::Write, ops::Range, vec::Vec};

#[derive(Clone, Debug)]
struct PathNode {
    neuron: usize,
    neuron_dependencies: Vec<usize>,
    input_dependencies: Vec<usize>,
}

#[derive(Debug)]
struct NeuralNetwork {
    neurons: Vec<Neuron>,
    weights: Vec<f64>,
    pub input_number: usize,
    paths: Vec<PathNode>,
}

#[derive(Debug)]
struct Neuron {
    connections: Vec<Connection>,
    bias: f64,
    activation: fn(f64) -> f64,
    #[allow(dead_code)]
    name: String,
}

#[derive(Debug)]
enum Connection {
    Neuron(usize, usize),
    Input(usize, usize),
}

impl NeuralNetwork {
    fn new(input_number: usize) -> NeuralNetwork {
        NeuralNetwork {
            neurons: Vec::new(),
            weights: Vec::new(),
            input_number,
            paths: Vec::new(),
        }
    }

    fn add_neuron(&mut self, bias: f64, activation: fn(f64) -> f64, name: String) -> usize {
        self.neurons.push(Neuron {
            connections: Vec::new(),
            bias,
            activation,
            name,
        });
        self.neurons.len() - 1
    }

    fn connect_neuron_to_neuron(&mut self, from: usize, to: usize, weight: f64) {
        let weight_index = self.weights.len();
        self.weights.push(weight);
        self.neurons[to]
            .connections
            .push(Connection::Neuron(from, weight_index));
    }

    fn connect_neuron_to_input(&mut self, from: usize, to: usize, weight: f64) {
        let weight_index = self.weights.len();
        self.weights.push(weight);
        self.neurons[to]
            .connections
            .push(Connection::Input(from, weight_index));
    }

    fn compute_path(&mut self) {
        // Step 1: Create path nodes
        let mut path_nodes = Vec::new();
        for (neuron_index, neuron) in self.neurons.iter().enumerate() {
            let mut neuron_dependencies = Vec::new();
            let mut input_dependencies = Vec::new();
            for connection in &neuron.connections {
                match connection {
                    Connection::Neuron(index, _) => neuron_dependencies.push(*index),
                    Connection::Input(index, _) => input_dependencies.push(*index),
                }
            }
            path_nodes.push(PathNode {
                neuron: neuron_index,
                neuron_dependencies,
                input_dependencies,
            });
        }

        // Step 2: Sort path nodes by dependencies (from input to not used neurons)
        let mut sorted_path_nodes: Vec<PathNode> = Vec::new();
        let mut added: Vec<bool> = vec![false; path_nodes.len()];

        fn add_node_to_sorted_path_nodes(
            path_nodes: &mut Vec<PathNode>,
            sorted_path_nodes: &mut Vec<PathNode>,
            added: &mut Vec<bool>,
            index: usize,
        ) -> usize {
            let node = path_nodes[index].clone();

            if node.neuron_dependencies.is_empty() && !node.input_dependencies.is_empty() {
                // This node is one of the first nodes
                sorted_path_nodes.insert(0, node);
                added[index] = true;
                0
            } else if !node.neuron_dependencies.is_empty() {
                // If the dependent neurons are already in the sorted list, we can add this node
                // to the sorted list
                // If not, we need to add the dependent neurons first
                // NOTE: This means that circular dependencies are not handled
                let mut last_index = 0;
                for dependency in &node.neuron_dependencies {
                    if let Some(i) = sorted_path_nodes
                        .iter()
                        .position(|n| n.neuron == *dependency)
                    {
                        if i > last_index {
                            last_index = i;
                        }
                    } else {
                        let i = add_node_to_sorted_path_nodes(
                            path_nodes,
                            sorted_path_nodes,
                            added,
                            *dependency,
                        );
                        if i > last_index && i != usize::MAX {
                            last_index = i;
                        }
                    }
                }

                sorted_path_nodes.insert(last_index + 1, node);
                added[index] = true;
                last_index + 1
            } else {
                // This node is not used
                added[index] = true;
                usize::MAX
            }
        }

        for index in 0..path_nodes.len() {
            if !added[index] {
                add_node_to_sorted_path_nodes(
                    &mut path_nodes,
                    &mut sorted_path_nodes,
                    &mut added,
                    index,
                );
                added[index] = true;
            }
        }

        self.paths = sorted_path_nodes;
    }

    #[allow(dead_code)]
    fn train(&mut self, training_data: Vec<(Vec<f64>, f64)>, epochs: usize) {
        for _ in 0..epochs {
            for (inputs, target) in &training_data {
                let output = self.forward(inputs);
                let error = target - output;
                self.backpropagate(error);
            }
        }
    }

    fn train_finite_difference(
        &mut self,
        training_data: &Vec<(Vec<f64>, f64)>,
        epochs: usize,
        epsilon: f64,
        learning_rate: f64,
    ) -> f64 {
        let mut variance = 0.0;
        let mut last_cost = 0.0;
        for _ in 0..epochs {
            let mut cost = 0.0;
            for (inputs, target) in training_data {
                let output = self.forward(inputs);
                let dcost = (target - output).powi(2);
                cost += dcost;
            }
            cost /= training_data.len() as f64;
            variance += (last_cost - cost).abs();

            self.backpropagate_finite_difference(&training_data, epsilon, learning_rate, cost);

            last_cost = cost;
        }

        variance / epochs as f64
    }

    pub fn cost_finite_difference(&self, training_data: &Vec<(Vec<f64>, f64)>) -> f64 {
        let mut cost = 0.0;
        for (inputs, target) in training_data {
            let output = self.forward(inputs);
            let dcost = (target - output).powi(2);
            cost += dcost;
        }
        cost /= training_data.len() as f64;

        cost
    }

    pub fn get_neuron_count(&self) -> usize {
        self.neurons.len()
    }

    pub fn compute_neuron(&self, neuron: usize, inputs: &Vec<f64>) -> f64 {
        let mut sum = 0.0;
        for connection in &self.neurons[neuron].connections {
            match connection {
                Connection::Neuron(index, weight_index) => {
                    sum += self.compute_neuron(*index, inputs) * self.weights[*weight_index]
                }
                Connection::Input(index, weight_index) => {
                    sum += inputs[*index] * self.weights[*weight_index]
                }
            }
        }
        (self.neurons[neuron].activation)(sum + self.neurons[neuron].bias)
    }

    fn forward(&self, inputs: &Vec<f64>) -> f64 {
        assert!(inputs.len() == self.input_number);

        let mut outputs = vec![0.0; self.neurons.len()];
        for path_node in &self.paths {
            outputs[path_node.neuron] = self.compute_neuron(path_node.neuron, inputs);
        }

        outputs[outputs.len() - 1]
    }

    fn backpropagate_finite_difference(
        &mut self,
        training_data: &Vec<(Vec<f64>, f64)>,
        epsilon: f64,
        learning_rate: f64,
        cost: f64,
    ) {
        // Step 1: Calculate the number of weights in the network
        let mut weight_number = 0;
        for neuron in &self.neurons {
            weight_number += neuron.connections.len();
        }

        // Step 2: Calculate the gradient of the cost function with respect to each weight
        let mut gradients = vec![0.0; weight_number];
        let mut bias_gradients = vec![0.0; self.neurons.len()];
        let mut i = 0;
        for j in 0..self.neurons.len() {
            let old_bias = self.neurons[j].bias;
            self.neurons[j].bias += epsilon;
            bias_gradients[j] = (self.cost_finite_difference(training_data) - cost) / epsilon;
            self.neurons[j].bias = old_bias;

            let neuron = &self.neurons[j];

            for connection in &neuron.connections {
                gradients[i] = match *connection {
                    Connection::Neuron(_, weight_index) => {
                        let old_weight = self.weights[weight_index];
                        self.weights[weight_index] += epsilon;
                        let gradient =
                            (self.cost_finite_difference(training_data) - cost) / epsilon;
                        self.weights[weight_index] = old_weight;
                        gradient
                    }
                    Connection::Input(_, weight_index) => {
                        let old_weight = self.weights[weight_index];
                        self.weights[weight_index] += epsilon;
                        let gradient =
                            (self.cost_finite_difference(training_data) - cost) / epsilon;
                        self.weights[weight_index] = old_weight;
                        gradient
                    }
                };
                i += 1;
            }
        }

        // Step 3: Update the weights
        i = 0;
        for (j, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.bias -= learning_rate * bias_gradients[j];
            for connection in &mut neuron.connections {
                match connection {
                    Connection::Neuron(_, weight_index) => {
                        self.weights[*weight_index] -= learning_rate * gradients[i];
                    }
                    Connection::Input(_, weight_index) => {
                        self.weights[*weight_index] -= learning_rate * gradients[i];
                    }
                }
                i += 1;
            }
        }
    }

    #[allow(dead_code)]
    fn backpropagate(&mut self, _error: f64) {}

    fn get_max_depth(&self, neuron: usize, current_depth: usize) -> usize {
        let mut max_depth = current_depth;
        for connection in &self.neurons[neuron].connections {
            match connection {
                Connection::Neuron(index, _) => {
                    let depth = self.get_max_depth(*index, current_depth + 1);
                    if depth > max_depth {
                        max_depth = depth;
                    }
                }
                _ => {}
            }
        }
        max_depth
    }

    fn dump(&self) {
        // Step 1: Dump the network architecture, layer by layer
        println!("| Network architecture:");
        // The inputs are in layer 0, a particular neuron is in layer 1 + the number of chained
        // dependencies
        let mut layers = Vec::new();
        for i in 0..self.neurons.len() {
            let depth = self.get_max_depth(i, 0);
            if depth >= layers.len() {
                layers.push(Vec::new());
            }
            layers[depth].push(i);
        }

        print!("| ");
        for i in 0..self.input_number {
            print!("Input {}    ", i);
        }

        for layer in layers.iter() {
            print!("\n| ");
            for neuron in layer {
                print!(
                    "Neuron {} (bias={})    ",
                    self.neurons[*neuron].name, self.neurons[*neuron].bias
                );
            }
        }
        println!("\n| Weights: ");

        // Step 2: Dump the weights
        // The weights are dumped with their corresponding connection
        for neuron in self.neurons.iter() {
            for connection in &neuron.connections {
                match connection {
                    Connection::Neuron(index, weight_index) => {
                        println!(
                            "| Neuron {} -> Neuron {} : {}",
                            neuron.name, self.neurons[*index].name, self.weights[*weight_index]
                        );
                    }
                    Connection::Input(index, weight_index) => {
                        println!(
                            "| Input {} -> Neuron {} : {}",
                            index, neuron.name, self.weights[*weight_index]
                        );
                    }
                }
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn round(x: f64) -> f64 {
    if x > 0.5 {
        1.0
    } else {
        0.0
    }
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn create_xor_nn(range: Range<f64>, activation: fn(f64) -> f64) -> NeuralNetwork {
    let mut rng = rand::thread_rng();
    let mut nn = NeuralNetwork::new(2);
    // Emulate the XOR function
    // The network should have 2 inputs, 2 neurons, then 1 neuron.
    let n11 = nn.add_neuron(rng.gen_range(range.clone()), activation, "n11".to_owned());
    nn.connect_neuron_to_input(0, n11, rng.gen_range(range.clone()));
    nn.connect_neuron_to_input(1, n11, rng.gen_range(range.clone()));

    let n12 = nn.add_neuron(rng.gen_range(range.clone()), activation, "n12".to_owned());
    nn.connect_neuron_to_input(0, n12, rng.gen_range(range.clone()));
    nn.connect_neuron_to_input(1, n12, rng.gen_range(range.clone()));

    let n21 = nn.add_neuron(rng.gen_range(range.clone()), activation, "n21".to_owned());
    nn.connect_neuron_to_neuron(n11, n21, rng.gen_range(range.clone()));
    nn.connect_neuron_to_neuron(n12, n21, rng.gen_range(range));

    nn.compute_path();

    nn
}

fn ask_neural_network_parameters() -> Result<NeuralNetwork, Box<dyn Error>> {
    let mut input = String::new();

    print!("| Enter the activation function to use (sigmoid, truncate or relu): ");
    std::io::stdout().flush().unwrap();
    std::io::stdin().read_line(&mut input).unwrap();
    let activation = match input.trim() {
        "sigmoid" => sigmoid,
        "truncate" => round,
        "relu" => relu,
        _ => return Err("Invalid activation function".into()),
    };

    print!("| Enter the range of the weights (start..end, with end exclusive): ");
    std::io::stdout().flush().unwrap();
    input.clear();
    std::io::stdin().read_line(&mut input).unwrap();
    let mut parts = input.trim().split("..");
    let start: f64 = parts
        .next()
        .ok_or("Invalid range")?
        .parse()
        .map_err(|_| "Invalid range")?;
    let end: f64 = parts
        .next()
        .ok_or("Invalid range")?
        .parse()
        .map_err(|_| "Invalid range")?;
    let range = start..end;

    Ok(create_xor_nn(range, activation))
}

fn main() {
    let training_data = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ];

    let mut nn: NeuralNetwork;

    // Initialize the neural network
    println!("|==========================|");
    loop {
        match ask_neural_network_parameters() {
            Ok(neural_network) => {
                nn = neural_network;
                println!("|==========================|");
                break;
            }
            Err(e) => println!("| {}", e),
        }
        println!("|==========================|");
    }

    loop {
        println!(
            "| Current cost: {}",
            nn.cost_finite_difference(&training_data)
        );
        println!("| What do you want to do?");
        println!("| 1. Train the neural network");
        println!("| 2. Test the neural network with custom inputs");
        println!("| 3. Test the neural network with the training data");
        println!("| 4. Reset the neural network");
        println!("| 5. Dump the neural network");
        println!("| 6. Quit");
        print!("| > ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        match input {
            "1" => {
                print!("| Enter the number of epochs: ");
                std::io::stdout().flush().unwrap();
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let epochs: usize = match input.trim().parse() {
                    Ok(epochs) => epochs,
                    Err(_) => {
                        println!("| Invalid number of epochs");
                        continue;
                    }
                };

                print!("| Enter epsilon: ");
                std::io::stdout().flush().unwrap();
                input.clear();
                std::io::stdin().read_line(&mut input).unwrap();
                let epsilon: f64 = match input.trim().parse() {
                    Ok(epsilon) => epsilon,
                    Err(_) => {
                        println!("| Invalid epsilon");
                        continue;
                    }
                };

                print!("| Enter the learning rate: ");
                std::io::stdout().flush().unwrap();
                input.clear();
                std::io::stdin().read_line(&mut input).unwrap();
                let learning_rate: f64 = match input.trim().parse() {
                    Ok(learning_rate) => learning_rate,
                    Err(_) => {
                        println!("| Invalid learning rate");
                        continue;
                    }
                };

                println!("| Training the neural network...");
                let variance =
                    nn.train_finite_difference(&training_data, epochs, epsilon, learning_rate);
                println!("| The variance with these parameters is: {}", variance);
            }
            "2" => {
                print!("| Enter the first input: ");
                std::io::stdout().flush().unwrap();
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let input1: f64 = match input.trim().parse() {
                    Ok(input1) => input1,
                    Err(_) => {
                        println!("| Invalid input");
                        continue;
                    }
                };

                print!("| Enter the second input: ");
                std::io::stdout().flush().unwrap();
                input.clear();
                std::io::stdin().read_line(&mut input).unwrap();
                let input2: f64 = match input.trim().parse() {
                    Ok(input2) => input2,
                    Err(_) => {
                        println!("| Invalid input");
                        continue;
                    }
                };

                println!("| The output is: {}", nn.forward(&vec![input1, input2]));
            }
            "3" => {
                let mut input = String::new();
                print!("| Should the output of hidden neurons be displayed? (y/n): ");
                std::io::stdout().flush().unwrap();
                std::io::stdin().read_line(&mut input).unwrap();
                let display_hidden = match input.trim() {
                    "y" | "Y" => true,
                    "n" | "N" => false,
                    _ => {
                        println!("| Invalid option");
                        continue;
                    }
                };

                println!("| Testing the neural network with the training data...");
                for (inputs, expected_output) in &training_data {
                    if !display_hidden {
                        let output = nn.forward(inputs);
                        println!(
                            "| Inputs: {:?}, Expected output: {}, Output: {}",
                            inputs, expected_output, output
                        );
                    } else {
                        println!("| Inputs: {:?}", inputs);
                        for i in 0..nn.get_neuron_count() {
                            println!("| Output of neuron {}: {}", i, nn.compute_neuron(i, inputs));
                        }

                        println!(
                            "| Expected output: {}, Output: {}",
                            expected_output,
                            nn.forward(inputs)
                        );
                    }
                }
            }
            "4" => {
                println!("| Resetting the neural network...");
                loop {
                    match ask_neural_network_parameters() {
                        Ok(neural_network) => {
                            nn = neural_network;
                            break;
                        }
                        Err(e) => println!("| {}", e),
                    }
                }
            }
            "5" => {
                println!("| Dumping the neural network...");
                nn.dump();
            }
            "6" => {
                println!("| Quitting...");
                break;
            }
            _ => {
                println!("| Invalid option");
            }
        }
        println!("|==========================|");
    }
}
