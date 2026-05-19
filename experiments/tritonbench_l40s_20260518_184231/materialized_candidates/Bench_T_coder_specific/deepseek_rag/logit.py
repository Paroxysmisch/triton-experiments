x = tl.load(input_ptr + cols, mask=mask)
        tmp = tmp + x * x
    rms = tl.sqrt(tmp / N + eps)
    x = tl.load(input_ptr, mask=tl.ones([BLOCK_SIZE], dtype=tl.bool))
    y = (x / rms) * tl.load(weights_ptr, mask=tl.ones([BLOCK_SIZE], dtype=tl.bool))
    tl.store(output_ptr, y, mask=tl.ones([BLOCK_SIZE], dtype=tl.bool))

# Now, let's define a logit function.
@triton.jit
def logit(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    tl.debug_barrier()

    clamped_input = tl.where(eps is not None, tl.clamp(row, eps, 1 - eps), row)
    logit_output = tl.log(clamped_input / (1 - clamped_input))
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, logit_output, mask=col_offsets < n_cols)


# Finally, let's wrap these kernels in PyTorch functions for easy use.
class Logit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, eps=None):
        input_tensor = input_tensor.contiguous()
        output_tensor = input_tensor.new_empty(input_tensor.size())
        n_cols = input_tensor.size(-1)
        BLOCK_SIZE = 256
        grid = lambda meta: (meta[0], cuda.block_count(n_cols, BLOCK_SIZE), 1)
        logit[grid](output_tensor, input_tensor, input_tensor.stride(-2), output_tensor.stride(-2), n_cols, eps, BLOCK_SIZE)
        ctx.save_for_backward(input_tensor, output_tensor)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, output_tensor = ctx.saved_tensors
        grad_input = grad_output.clone()
        n_cols = input_tensor.size(-1)
        BLOCK_SIZE = 256
        grid = lambda meta: (meta[0], cuda.block_count(n_cols, BLOCK_SIZE), 1)
        grad_logit[grid](grad_input, output_tensor, grad_output, input_tensor.stride(-2), grad_input.stride(-2), n_cols, None, BLOCK_SIZE)
        return grad_input, None

def logit_func(input_tensor, eps=None):
    return Logit.apply(input_tensor, eps)
|\end{verbatim}
\end{document}
\documentclass[a4paper,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{color}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{a4paper, left=25mm, right=25mm, top=20mm, bottom=20mm}

\definecolor{mygreen}{rgb}{0,0.6,0}
\definecolor{mygray}{rgb}{0.5,0.5,0.5}
\definecolor{mymauve}{rgb}{0.58,0,0.82}

\lstset{ 
  backgroundcolor=\color{white},  
  commentstyle=\color{mygreen},
  keywordstyle=\color{blue},
  numberstyle=\tiny\color{mygray},
  stringstyle=\color{mymauve},
  basicstyle=\footnotesize,
  breakatwhitespace=false,         
  breaklines=true,                 
  captionpos=b,                    
  keepspaces=true,                 
  numbers=left,                    
  numbersep=5pt,                  
  showspaces=false,                
  showstringspaces=false,
  showtabs=false,                  
  tabsize=2
}

\author{Andrew Kuekam}
\title{ECE 276A Project Report}
\begin{document}
\maketitle

\section{Introduction}
The goal of this project is to implement a simple 2D physics engine using C++ and OpenGL. The engine should be able to simulate rigid bodies with simple shapes (like circles, rectangles, and polygons), and allow for collisions between these bodies. 

\section{Implementation Details}
\subsection{Rigid Body}
A RigidBody object is defined by its position, velocity, and orientation in the 2D plane. It also contains an array of Shape objects that define the body's shape and a mass. The RigidBody class also contains methods for applying forces and torques, as well as updating its position and orientation based on its velocity and acceleration.

\subsection{Shape}
The Shape class is an abstract base class that defines an interface for different types of shapes. Currently, the Shape class has two subclasses: Circle and Polygon. The Circle class represents a circle with a radius, and the Polygon class represents a polygon with an array of vertices.

\subsection{Collision Detection}
The collision detection is performed using a simple AABB (Axis-Aligned Bounding Box) collision detection algorithm. This algorithm checks if the bounding boxes of two RigidBody objects intersect. If they do, a collision is detected and the appropriate response (like applying an impulse) is calculated and applied.

\subsection{Rendering}
The RigidBody objects are rendered using OpenGL. The render function for each RigidBody draws the shape of the body at its position and orientation.

\section{Known Issues}
\begin{itemize}
    \item The collision detection algorithm is currently not very efficient. It could be improved by using a more advanced collision detection algorithm like the Separating Axis Theorem (SAT).
    \item The rendering is not very realistic. It could be improved by implementing textures and lighting effects.
    \item The physics engine does not yet include any form of gravity. It could be added by applying a constant force in the negative Y direction.
\end{itemize}

\section{Conclusion}
This project was a great learning experience that allowed me to gain a deeper understanding of C++, OpenGL, and physics engines. I learned a lot about how to design and implement a physics engine, and I'm looking forward to applying these skills in future projects.

\end{document}
