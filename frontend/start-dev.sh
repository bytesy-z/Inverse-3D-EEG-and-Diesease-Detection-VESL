#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
cd /data1tb/VESL/fyp-2.0/frontend
exec npx next dev -p 3000
