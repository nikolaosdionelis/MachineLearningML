gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)
out = netG(gen_input)
noise_eta = torch.randn_like(out)
g_fake_data = out + noise_eta * sigma_x

dg_fake_decision = netD(g_fake_data)
g_error_gan = criterion(dg_fake_decision, label)
D_G_z2 = dg_fake_decision.mean().item()

if args.lambda_ == 0:
    g_error_gan.backward()
    optimizerG.step()
    sigma_optimizer.step()

else:
    hmc_samples, acceptRate, stepsize = hmc.get_samples(
        netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
        args.hmc_learning_rate, args.hmc_opt_accept)

    bsz, d = hmc_samples.size()
    mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
    bsz = g_fake_data.size(0)

    mean_output_summed = torch.zeros_like(g_fake_data)
    for cnt in range(args.num_samples_posterior):
        mean_output_summed = mean_output_summed + mean_output[cnt * bsz:(cnt + 1) * bsz]
    mean_output_summed = mean_output_summed / args.num_samples_posterior

    c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
    g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()