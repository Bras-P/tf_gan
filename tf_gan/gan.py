import tensorflow as tf


class GAN:

    def __init__(
        self,
        gen_network,
        disc_network,
        generate_latent_variables,
        gen_opt = tf.keras.optimizers.Adam(2e-4),
        disc_opt = tf.keras.optimizers.Adam(2e-4),
    ):
        self.gen_network = gen_network
        self.disc_network = disc_network
        self.generate_latent_variables = generate_latent_variables
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt


    def gen_loss(self, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def disc_loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    

    def train_step(self, data_batch):
        batch_size = len(data_batch)
        latent_variable = self.generate_latent_variables(batch_size)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.gen_network(latent_variable, training=True)
            real_output = self.disc_network(data_batch, training=True)
            fake_output = self.disc_network(generated_data, training=True)
            loss_gen = self.gen_loss(fake_output)
            loss_disc = self.disc_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(loss_gen, self.gen_network.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(loss_disc, self.disc_network.trainable_variables)

        self.gen_opt.apply_gradients(zip(gradients_of_generator, self.gen_network.trainable_variables))
        self.disc_opt.apply_gradients(zip(gradients_of_discriminator, self.disc_network.trainable_variables))

        return loss_gen.numpy(), loss_disc.numpy()

    
    def train(self, dataset, epochs):
        self.gen_losses = []
        self.disc_losses = []

        for epoch in range(epochs):
            loss_gen, loss_disc = 0., 0.

            for data_batch in dataset:
                loss_gen_new, loss_disc_new = self.train_step(data_batch)
                loss_gen += loss_gen_new
                loss_disc += loss_disc_new
            
            self.gen_losses.append(loss_gen/len(dataset))
            self.disc_losses.append(loss_disc/len(dataset))

            print('Epoch {}, loss: {}\n'.format(epoch, loss_gen + loss_disc))

    
    def generate_samples(self, batch_size):
        latent_variable = self.generate_latent_variables(batch_size)
        return self.gen_network(latent_variable, training=False)

    


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# since we use from_logits=True, do not use sigmoid in the disc_network