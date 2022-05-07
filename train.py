def train(do_save_model = False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = data_loaded
    current_step = 0
    
    for epoch in range(EPOCHS):
        for real_A, real_B in tqdm(dataloader):
            curr_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_AB(real_B)
            disc_A_loss = get_discriminator_loss(real_A, fake_A, disc_A, adversarial_criterion)
            disc_A_loss.backward(retain_graph = True)
            disc_A_opt.step()
            
            disc_B_opt.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_discriminator_loss(real_B, fake_B, disc_B, adversarial_criterion)
            disc_B_loss.backward(retain_graph = True)
            disc_B_opt.step()
            
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_generator_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adversarial_criterion, recon_criterion, recon_criterion)
            gen_loss.backward(retain_graph = True)
            gen_opt.step()
            
            mean_discriminator_loss += disc_A_loss.item() / DISPLAY_STEP
            mean_generator_loss += gen_loss.item() / DISPLAY_STEP
            
            ## Code for Visualization
            if current_step % DISPLAY_STEP == 0:
                print(f"Epoch {epoch}: Step {current_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if do_save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"cycleGAN_{current_step}.pth")
            current_step += 1
            
            