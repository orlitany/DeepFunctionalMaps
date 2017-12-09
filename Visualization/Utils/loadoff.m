function shape = loadoff_colorFix(filename)
% This function reads *.off files.
% Added: reading color per vertex.
%
% Input: filename (including full path and ext)
% Output:
% shape - a struct containing: x,y,z values and TRIV (if exist)
% written by Or Litany (orlitany <at> gmail <dot> com )%
% added another color-fix

shape = [];

f = fopen(filename, 'rt');

txt = fgetl(f);
if strcmpi(txt,'COFF'),
    line_entries = 7;
elseif strcmpi(txt,'OFF'),
    line_entries = 3;
else
    disp('unknown file type');
end


n = sscanf(fgetl(f), '%d %d %d');
nv = n(1);
nt = n(2);

data = fscanf(f, '%f');

shape.TRIV = reshape(data(line_entries*nv+1:line_entries*nv+4*nt), [4 nt])';
shape.TRIV = shape.TRIV(:,2:end) + 1;

data = data(1:line_entries*nv);
data = reshape(data, [line_entries nv]);


% shape.X = data(1,:)';
% shape.Y = data(2,:)';
% shape.Z = data(3,:)';
shape.VERT = [data(1,:)' data(2,:)' data(3,:)'];

if strcmpi(txt,'COFF'),
    color_per_vertex = data(4:6,:)';
    shape.color = color_per_vertex;
end


fclose(f);
