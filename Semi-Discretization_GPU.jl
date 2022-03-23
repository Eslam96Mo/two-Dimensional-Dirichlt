using CUDA
using CairoMakie

α  = 1e-4                                                  # Diffusivity
L  = 0.1                                                   # Length
W  = 0.1                                                   # Width
Nx = 66                                                    # No.of steps in x-axis
Ny = 66                                                    # No.of steps in y-axis
Δx = L/(Nx-1)                                               # x-grid spacing
Δy = W/(Ny-1)                                               # y-grid spacing
Δt = Δx^2 * Δy^2 / (2.0 * α * (Δx^2 + Δy^2))               # Largest stable time step

temp_left   = 0                                        # Boundary condition
temp_right  = 0
temp_bottom = 0
temp_top    = 0

function diffuse!(du, u, p,t)
      dijij = view(du, 2:Nx-1, 2:Ny-1)
      dij  = view(u, 2:Nx-1, 2:Ny-1)
      di1j = view(u, 1:Nx-2, 2:Ny-1)
      dij1 = view(u, 2:Nx-1, 1:Ny-2)
      di2j = view(u, 3:Nx  , 2:Ny-1)
      dij2 = view(u, 2:Nx-1, 3:Ny  )                                                  # Stencil Computations
  
      @. dijij  = α  * ( (di1j - 2 * dij + di2j)/Δx^2 + (dij1 - 2 * dij + dij2)/Δy^2)                                               # Apply diffusion

    du[1, :]    .= temp_left 
    du[Nx, :]   .= temp_right 
    du[:, 1]    .= temp_bottom 
    du[:, Ny]   .= temp_top                              # update boundary condition (Dirichlet BCs) 
    
end


u_GPU = CUDA.zeros(Nx,Ny)   
u_GPU[28:38, 28:38] .= 5                                                      

fig, pltpbj = plot(u_GPU; colormap  = :viridis ,markersize = 5, linestyle = ".-", 
figure = (resolution = (600, 400), font = "CMU Serif"),
        axis =  ( xlabel ="Grid points (Nx)", ylabel ="Grid points (Ny)", backgroundcolor = :white,
        xlabelsize = 15, ylabelsize = 15))
        Colorbar(fig[1,2], limits = (0, 5),label = "Heat conduction") 
display(fig)

using DifferentialEquations, DiffEqGPU

tspan = (0.0, 1.0)
prob = ODEProblem(diffuse!, u_GPU, tspan)
sol = solve(prob, Euler(), dt=Δt)

for i =  2 : length(sol)
   fig , pltpbj = plot(sol.u[i]; colormap  = :viridis ,markersize = 5, linestyle = ".-", 
   figure = (resolution = (600, 400), font = "CMU Serif"),
       axis =  ( xlabel ="Grid points (Nx)", ylabel ="Grid points (Ny)", backgroundcolor = :white,
       xlabelsize = 15, ylabelsize = 15))
       Colorbar(fig[1,2], limits = (0, 5),label = "Heat conduction")
   display(fig)
end