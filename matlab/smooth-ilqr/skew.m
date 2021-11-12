function skew = skew(w)
a1 = w(1);
a2 = w(2);
a3 = w(3);
skew = [0 -a3 a2; a3 0 -a1; -a2 a1 0];
end